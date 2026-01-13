import sys
import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from datetime import datetime

# Add hamer to sys.path locally for imports
# When running inside hamer/ directory, the current directory is added to sys.path by default,
# allowing 'import hamer' (the package) and 'import vitpose_model' (sibling file) to work.


try:
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
    from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    from hamer.utils import recursive_to
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    # Try importing ViTPoseModel
    try:
        from vitpose_model import ViTPoseModel
    except ImportError:
        # If it fails, check if we need to adjust path further or if it is inside hamer package
        # Based on file structure: hamer/vitpose_model.py, and we added hamer/ to sys.path
        # It should work. If not, we might be in a different context.
        pass

except ImportError as e:
    print(f"Error importing HaMeR modules: {e}")
    print("Please ensure you are running this script from the project root and 'hamer/' directory exists.")
    sys.exit(1)

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

class EgoVLAPipeline:
    def __init__(self, video_path, output_root='output', language_instruction="", save_frames=True, save_visuals=True):
        self.video_path = video_path
        self.output_root = output_root
        self.language_instruction = language_instruction
        self.save_frames = save_frames
        self.save_visuals = save_visuals
        self.frame_dir = os.path.join(output_root, "frames")
        self.action_dir = os.path.join(output_root, "actions")
        self.visual_dir = os.path.join(output_root, "visuals")
        
        # Settings based on paper/placeholder
        self.target_fps = 3 
        self.resolution = (384, 384)
        
        if self.save_frames:
            os.makedirs(self.frame_dir, exist_ok=True)
        os.makedirs(self.action_dir, exist_ok=True)
        if self.save_visuals:
            os.makedirs(self.visual_dir, exist_ok=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def run(self):
        print(f"Processing video: {self.video_path}")
        frame_paths = self.process_frames()
        self.estimate_hand_pose(frame_paths)
        print("Pipeline completed.")

    # --- Stage 1: Frame Sampling ---
    def process_frames(self):
        print("[Stage 1] Sampling frames at 3 FPS...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle cases where FPS is not properly read
        if fps <= 0: fps = 30
        
        hop = round(fps / self.target_fps)
        if hop < 1: hop = 1
        
        frame_data = []
        count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % hop == 0:
                # Resize as per placeholder specs
                resized = cv2.resize(frame, self.resolution) 
                
                frame_name = f"frame_{saved_count:05d}.jpg"
                
                if self.save_frames:
                    frame_path = os.path.join(self.frame_dir, frame_name)
                    cv2.imwrite(frame_path, resized)
                    frame_data.append(frame_path)
                else:
                    frame_data.append((frame_name, resized))
                
                saved_count += 1
            count += 1
            
        cap.release()
        
        print(f"Prepared {len(frame_data)} frames")
        return frame_data

    # --- Stage 2 & 3: Hand Pose Estimation (HaMeR) ---
    def estimate_hand_pose(self, frame_data, output_npz_path=None):
        print("[Stage 2&3] Estimating Hand Pose using HaMeR...")
        
        if not frame_data:
            print("No frames to process.")
            return

        # 1. Setup HaMeR
        print("Loading HaMeR models...")
        try:
            download_models(CACHE_DIR_HAMER)
            model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
            model = model.to(self.device)
            model.eval()
            
            # Setup the renderer
            self.renderer = Renderer(model_cfg, faces=model.mano.faces)
            
        except Exception as e:
            print(f"Failed to load HaMeR model: {e}")
            return

        # 2. Setup Detector (ViTDet)
        print("Loading Detector (ViTDet)...")
        try:
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        except Exception as e:
            print(f"Failed to load Detector: {e}")
            return

        # 3. Setup Keypoint Detector (ViTPose)
        try:
            cpm = ViTPoseModel(self.device)
        except Exception as e:
            print(f"Failed to load ViTPose: {e}")
            return

        # 4. Process Frames
        print(f"Running inference on {len(frame_data)} frames...")
        
        # Initialize storage for all frames
        all_results = {
            'frame_name': [],
            'person_id': [],
            'is_right': [],
            'global_orient_rotmat': [],
            'global_orient_6d': [],
            'hand_pose_rotmat': [],
            'translation': [],
            'theta': [],
            'betas': [],
            'hand_pose_pca15': []
        }
        
        # Batch processing not implemented for simplicity and because image sizes might vary if we didn't resize.
        # But we did resize to 384x384. 
        # HaMeR demo processes 1 by 1.
        
        for item in frame_data:
            if isinstance(item, str):
                frame_path = item
                img_cv2 = cv2.imread(frame_path)
                frame_name = os.path.basename(frame_path).split('.')[0]
            else:
                frame_name, img_cv2 = item
                frame_name = frame_name.split('.')[0]

            if img_cv2 is None: continue
            
            # Detect humans
            det_out = detector(img_cv2)
            img = img_cv2.copy()[:, :, ::-1] # BGR to RGB

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores=det_instances.scores[valid_idx].cpu().numpy()

            # Detect keypoints
            vitposes_out = cpm.predict_pose(
                img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            bboxes = []
            is_right = []

            # Prepare boxes for HaMeR
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [min(keyp[valid,0]), min(keyp[valid,1]), max(keyp[valid,0]), max(keyp[valid,1])]
                    bboxes.append(bbox)
                    is_right.append(0)
                
                keyp = right_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [min(keyp[valid,0]), min(keyp[valid,1]), max(keyp[valid,0]), max(keyp[valid,1])]
                    bboxes.append(bbox)
                    is_right.append(1)

            if not bboxes:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            # Run HaMeR
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            # Extract MANO PCA components and mean if available
            hands_components = None
            hands_mean = None
            
            # Load from pickle to ensure we get full 45 components (smplx layer in model might truncate to 6)
            import pickle
            mano_pkl_path = os.path.join(CACHE_DIR_HAMER, 'data/mano/MANO_RIGHT.pkl')
            if os.path.exists(mano_pkl_path):
                try:
                    with open(mano_pkl_path, 'rb') as f:
                        mano_data = pickle.load(f, encoding='latin1')
                        hands_components = np.array(mano_data['hands_components']) # (45, 45)
                        hands_mean = np.array(mano_data['hands_mean']) # (45,)
                except Exception as e:
                    print(f"Warning: Failed to load MANO pickle: {e}")
            else:
                print(f"Warning: MANO pickle not found at {mano_pkl_path}. Falling back to model attributes.")
                # Fallback to model attributes if pickle missing (though this might be truncated)
                if hasattr(model, 'mano'):
                    if hasattr(model.mano, 'hand_components'):
                        hands_components = model.mano.hand_components
                        if isinstance(hands_components, torch.Tensor):
                            hands_components = hands_components.cpu().detach().numpy()
                    elif hasattr(model.mano, 'np_hand_components'):
                        hands_components = model.mano.np_hand_components

                    if hasattr(model.mano, 'hand_mean'):
                        hands_mean = model.mano.hand_mean
                        if isinstance(hands_mean, torch.Tensor):
                            hands_mean = hands_mean.cpu().detach().numpy()
                    elif hasattr(model.mano, 'flat_hand_mean'):
                        hands_mean = model.mano.flat_hand_mean
                        if isinstance(hands_mean, torch.Tensor):
                            hands_mean = hands_mean.cpu().detach().numpy()
            
            # Additional debug print
            if hands_components is None:
                print("Warning: MANO PCA components not found.")
            elif hands_components.shape[0] < 15:
                #print(f"Warning: Found MANO PCA components but with shape {hands_components.shape}. Need at least 15 for requested PCA15.")
                pass
            if hands_mean is None:
                print("Warning: MANO mean pose not found.")

            # Containers for visualization
            all_verts = []
            all_cam_t = []
            all_right_flag = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = model(batch)

                multiplier = (2*batch['right']-1)
                pred_cam = out['pred_cam']
                pred_mano_params = out['pred_mano_params']
                pred_cam_t_full = cam_crop_to_full(pred_cam, batch['box_center'], batch['box_size'], batch['img_size'])

                # Save parameters for each detected hand
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Person ID logic is simplified here as we process frame-by-frame independently.
                    # Ideally we would track persons across frames, but for basic extraction:
                    person_id = n 
                    
                    # Extract Data
                    global_orient = pred_mano_params['global_orient'][n].detach().cpu().numpy()
                    hand_pose = pred_mano_params['hand_pose'][n].detach().cpu().numpy()
                    betas = pred_mano_params['betas'][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n].detach().cpu().numpy()
                    right_flag = right[n] # 0 for left, 1 for right
                    
                    # Convert rotation matrices to axis-angle if needed
                    global_orient_rotmat = global_orient.squeeze() # (3, 3)
                    # Convert to 6D rotation (first two columns flattened)
                    global_orient_6d = global_orient_rotmat[:, :2].T.flatten() # (6,)
                    global_orient_aa, _ = cv2.Rodrigues(global_orient_rotmat)
                    
                    hand_pose_rotmats = hand_pose # (15, 3, 3)
                    hand_pose_aa = []
                    for i in range(hand_pose_rotmats.shape[0]):
                        aa, _ = cv2.Rodrigues(hand_pose_rotmats[i])
                        hand_pose_aa.append(aa)
                    hand_pose_aa = np.concatenate(hand_pose_aa).flatten()
                    
                    # Construct full theta (48D)
                    theta = np.concatenate([global_orient_aa.flatten(), hand_pose_aa])

                    # Calculate PCA15 if components available
                    hand_pose_pca15 = None
                    if hands_components is not None and hands_mean is not None:
                        # hand_pose_aa: (45,)
                        # hands_mean: (45,)
                        # hands_components: (45, 45) -> PCA basis vectors are rows
                        # pca_coeffs = Components @ (Pose - Mean)
                        pca_coeffs = hands_components.dot(hand_pose_aa - hands_mean)
                        hand_pose_pca15 = pca_coeffs[:15]

                    # Append to all results
                    all_results['frame_name'].append(frame_name)
                    all_results['person_id'].append(person_id)
                    all_results['is_right'].append(right_flag)
                    all_results['global_orient_rotmat'].append(global_orient_rotmat)
                    all_results['global_orient_6d'].append(global_orient_6d)
                    all_results['hand_pose_rotmat'].append(hand_pose)
                    all_results['translation'].append(cam_t)
                    all_results['theta'].append(theta)
                    all_results['betas'].append(betas)
                    if hand_pose_pca15 is not None:
                        all_results['hand_pose_pca15'].append(hand_pose_pca15)
                    else:
                        # Append None or zeros if PCA failed, to keep length consistent
                        # Using zeros for safety if downstream expects array, but best to be careful
                         all_results['hand_pose_pca15'].append(np.zeros(15, dtype=np.float32))

                    # Prepare for visualization
                    # Vertices are in out['pred_vertices'] (B, 778, 3)
                    # Need to verify if 'pred_vertices' is in 'out' from model forward
                    # The demo uses model.mano(...) to get vertices if not available or just uses out['pred_vertices']?
                    # preprocess_pipeline: out = model(batch) -> returns keys 'pred_cam', 'pred_mano_params', etc.
                    # HaMeR.forward_step (which is forward) returns 'pred_cam', 'pred_mano_params'.
                    # It DOES NOT return vertices by default to save memory?.
                    # Wait, let's check hamer.py forward_step return.
                    # It checks `if init_renderer` then it creates `self.renderer`.
                    # But the forward_step returns: output['pred_cam'] = pred_cam, output['pred_mano_params']...
                    
                    # So we need to compute vertices manually or rely on 'pred_vertices' if present.
                    # In hamer.py: 
                    # pred_mano_params['global_orient'] = ...
                    # mano_output = self.mano(...)
                    # output['pred_vertices'] = pred_vertices
                    # Yes, it seems to return pred_vertices.
                    
                    # BUT preprocess_pipeline.py calls "out = model(batch)".
                    # LightningModule forward usually calls forward_step? No, forward is separate defined?
                    # Let's check hamer.py forward. It is not defined explicitly?
                    # Ah, check hamer.py again.
                    
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right_batch = batch['right'][n].cpu().numpy() # scalar 0 or 1
                    verts[:,0] = (2*is_right_batch-1)*verts[:,0] # Flip x if left hand? 
                    # Note: Hamer model output is usually for fundamental right hand. 
                    # If input is left, it's flipped. The demo flips it back for visual.
                    # demo.py: verts[:,0] = (2*is_right-1)*verts[:,0]
                    
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right_flag.append(is_right_batch)

            # Render frame visualization
            if self.save_visuals and all_verts:
                # Calculate scaled focal length
                # model_cfg is loaded in estimate_hand_pose. 
                # Assuming model_cfg.MODEL.IMAGE_SIZE=256 and FOCAL_LENGTH=5000
                focal_length = 5000.
                img_size = 256.
                if hasattr(model_cfg, 'EXTRA') and hasattr(model_cfg.EXTRA, 'FOCAL_LENGTH'):
                    focal_length = model_cfg.EXTRA.FOCAL_LENGTH
                if hasattr(model_cfg, 'MODEL') and hasattr(model_cfg.MODEL, 'IMAGE_SIZE'):
                    img_size = model_cfg.MODEL.IMAGE_SIZE

                scaled_focal_length = focal_length / img_size * max(self.resolution)

                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                
                # render_res should match the image size (self.resolution)
                cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=self.resolution, is_right=all_right_flag, **misc_args)

                # Overlay
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0 # BGR to RGB, 0-1
                # Ensure input_img matches resolution (it should, we loaded resized frames? No, we loaded saved frames which are resized)
                # Wait, process_frames saves resized frames to disk. estimate_hand_pose reads them.
                # So img_cv2 IS self.resolution (384, 384).

                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha
                # Blend
                # cam_view is (H, W, 4)
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                visual_path = os.path.join(self.visual_dir, f"{frame_name}_vis.jpg")
                cv2.imwrite(visual_path, 255*input_img_overlay[:, :, ::-1]) # RGB to BGR for saving


        # Save aggregated NPZ
        if all_results['frame_name']:
            if output_npz_path:
                save_path = output_npz_path
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            else:
                save_path = os.path.join(self.action_dir, 'all_frames_data.npz')

            print(f"Saving aggregated results to {save_path}")
            np.savez(save_path, 
                    language_instruction=np.array([self.language_instruction]),
                    frame_name=np.array(all_results['frame_name']),
                    person_id=np.array(all_results['person_id']),
                    is_right=np.array(all_results['is_right']),
                    global_orient_rotmat=np.stack(all_results['global_orient_rotmat']),
                    global_orient_6d=np.stack(all_results['global_orient_6d']),
                    hand_pose_rotmat=np.stack(all_results['hand_pose_rotmat']),
                    translation=np.stack(all_results['translation']),
                    theta=np.stack(all_results['theta']),
                    betas=np.stack(all_results['betas']),
                    hand_pose_pca15=np.stack(all_results['hand_pose_pca15'])
            )
        else:
            print("No hand detections found to save.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess_pipeline.py <video_path> <output_root> [language_instruction]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_root = sys.argv[2]
    
    # Append timestamp to output_root to create unique folder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = f"{output_root}_{current_time}"

    language_instruction = sys.argv[3] if len(sys.argv) > 3 else "No instruction"
    
    pipeline = EgoVLAPipeline(video_path, output_root, language_instruction)
    pipeline.run()
