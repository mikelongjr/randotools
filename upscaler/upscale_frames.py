#!/usr/bin/env python3
import os
import sys
import threading
import torch
from PIL import Image
import grp, getpass
from queue import Queue
import torchvision

# Fix: Monkey-patch torchvision for basicsr compatibility (torchvision >= 0.13 removed functional_tensor)
# The `basicsr` library expects a module that was removed in torchvision >= 0.13.
# This patch intercepts the import error and shims the old module path to point to the new one.
try:
    import torchvision.transforms.functional_tensor
except ModuleNotFoundError:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F

# Fix for ModuleNotFoundError: No module named 'realesrgan.model'
# This can happen if a local/vendored library (py_real_esrgan) still refers to
# its original package name ('realesrgan') in its internal imports.
# We can fix this by aliasing the local module to the name it expects.
# import realesrgan
# help(realesrgan.models)
# sys.exit()
# sys.modules['realesrgan'] = sys.modules['py_real_esrgan']

# Fix: huggingface_hub >= 0.16 removed cached_download which py_real_esrgan uses.
# Patch it back as an alias for hf_hub_download before importing py_real_esrgan.
try:
    from huggingface_hub import cached_download  # noqa: F401
except ImportError:
    import huggingface_hub
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

from py_real_esrgan.model import RealESRGAN

# from realesrgan.models import RealESRGAN
# import realesrgan
# import realesrgan.models as RealESRGAN
# import realesrgan as RealESRGAN
from glob import glob
from tqdm import tqdm
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    RRDBNet = None
    print("\n[WARNING] 'basicsr' is not installed. Native x2 and Anime 6B models will be disabled.")
    print("          To fix on Fedora (skipping extension compilation):")
    print("            sudo dnf install gcc-c++ python3-devel")
    print("            CUDA_VISIBLE_DEVICES='' pip install basicsr --no-build-isolation\n")

# --- Configuration ---
INPUT_DIR = 'frames'
_OUTPUT_BASE = 'upscaled_frames'  # model suffix appended at runtime → e.g. upscaled_frames_x4plus
# ---------------------

def run_diagnostics():
    """Check for common ROCm/AMD GPU issues on Linux."""
    if sys.platform != "linux" or torch.cuda.is_available():
        return

    print("\n" + "="*40)
    print("AMD GPU DIAGNOSTICS")
    print("="*40)
    print(f"PyTorch version: {torch.__version__}")
    hip_ver = getattr(torch.version, 'hip', None)
    cuda_ver = getattr(torch.version, 'cuda', None)
    print(f"ROCm/HIP version: {hip_ver or 'N/A'}")

    # Detect the most common AMD GPU setup failure: the CUDA build of PyTorch
    # is installed on a machine with AMD GPU hardware.  The environment variable
    # overrides (HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES) have NO effect
    # with a CUDA-build torch — a ROCm-build torch must be installed first.
    if cuda_ver and not hip_ver:
        from upscaler.gpu_manager import GPUManager  # type: ignore
        if GPUManager._amd_hardware_present():
            print("")
            print("!" * 40)
            print("CRITICAL: Wrong PyTorch build!")
            print(f"  Installed : torch {torch.__version__}  (CUDA build — for NVIDIA)")
            print("  Required  : ROCm build  (e.g. +rocm6.2)")
            print("")
            print("  AMD GPUs CANNOT be used with a CUDA-build PyTorch.")
            print("  No environment variable can fix this.")
            print("")
            print("  Fix — reinstall PyTorch with ROCm support:")
            print("    pip install torch torchvision \\")
            print("      --index-url https://download.pytorch.org/whl/rocm6.2")
            print("  Or re-run:  ./fedora_setup.sh --amd")
            print("!" * 40)
            print("")
    
    # Check Group Permissions
    user = getpass.getuser()
    try:
        groups = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
        # Also check primary group
        gid = os.getgid()
        groups.append(grp.getgrgid(gid).gr_name)
        
        for required in ['render', 'video']:
            if required not in groups:
                print(f"CRITICAL: User '{user}' is NOT in the '{required}' group.")
                print(f"  Fix: sudo usermod -aG render,video {user} && reboot")
    except Exception as e:
        print(f"Could not check groups: {e}")

    # Check for device nodes
    has_kfd = os.path.exists("/dev/kfd")
    print(f"Kernel Device (/dev/kfd): {'Found' if has_kfd else 'NOT FOUND'}")

    if has_kfd and not os.access("/dev/kfd", os.R_OK | os.W_OK):
        print("PERMISSION ERROR: Current user cannot access /dev/kfd.")

    # Check for DRI devices
    dri_path = "/dev/dri"
    if os.path.exists(dri_path):
        print(f"DRI Devices: {os.listdir(dri_path)}")

    # Detect hardware GFX version(s) for all AMD GPU nodes in the KFD topology.
    # On laptops with an iGPU + discrete GPU (e.g. HP Omen) there are multiple
    # nodes; the CPU-only node (simd_count == 0) is skipped automatically.
    topology_root = "/sys/class/kfd/kfd/topology/nodes"
    if os.path.exists(topology_root):
        try:
            node_ids = sorted(
                int(n) for n in os.listdir(topology_root) if n.isdigit()
            )
        except Exception:
            node_ids = []
        gpu_index = 0  # HIP_VISIBLE_DEVICES uses GPU-only ordinals (0, 1, ...)
        for node_id in node_ids:
            props_path = os.path.join(topology_root, str(node_id), "properties")
            try:
                props = {}
                with open(props_path) as f:
                    for line in f:
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            props[parts[0]] = parts[1]
                if props.get("simd_count", "0") == "0":
                    continue
                ver = props.get("gfx_target_version", "unknown")
                print(f"Hardware GFX Version (node {node_id}, HIP device {gpu_index}): {ver}")
                if ver.isdigit() and len(ver) >= 6:
                    v = int(ver)
                    override = f"{v // 10000}.{(v // 100) % 100}.{v % 100}"
                    print(f"  Note: This GPU may need HSA_OVERRIDE_GFX_VERSION={override}")
                    print(f"        HSA_OVERRIDE_GFX_VERSION={override} HIP_VISIBLE_DEVICES={gpu_index} python upscale_frames.py")
                gpu_index += 1
            except Exception:
                pass
    else:
        print("Hardware GFX Version: Could not detect (KFD topology not found).")
    
    print("="*40 + "\n")
run_diagnostics()
def loader_thread(image_paths, q_in, num_workers, output_dir):
    """Loads image paths into the input queue."""
    for path in image_paths:
        basename = os.path.basename(path)
        output_path = os.path.join(output_dir, basename)
        q_in.put((path, output_path))
    # Add sentinels for workers
    for _ in range(num_workers):
        q_in.put(None)

def processor_thread(device, q_in, q_out):
    """Processes images from a queue on a specific GPU."""
    print(f"Processor thread started for device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'MPS'})")
    
    # Load the model for this specific device
    model = RealESRGAN(device, scale=4)
    anime_weights = 'weights/RealESRGAN_x4plus_anime_6B.pth'
    if os.path.exists(anime_weights) and RRDBNet:
        print(f"[{device}] Using fast Anime 6B model: {anime_weights}")
        model.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model.model.to(device)
        model.load_weights(anime_weights, download=False)
    else:
        print(f"[{device}] Using standard x4plus model (High Quality/Slower)")
        model.load_weights('weights/RealESRGAN_x4plus.pth', download=False)

    model.model.half()

    with torch.inference_mode():
        while True:
            item = q_in.get()
            if item is None:
                break
            
            path_to_image, output_path = item
            try:
                image = Image.open(path_to_image).convert('RGB')
                sr_image = model.predict(image)
                q_out.put((sr_image, output_path))
            except Exception as e:
                print(f"\n[{device}] Error processing {path_to_image}: {e}")
                # Signal a failure to saver so progress bar isn't stuck
                q_out.put((None, path_to_image))
    
    # Signal saver that this processor is done
    q_out.put(None)

def saver_thread(q_out, total_images, num_workers):
    """Saves images from a queue and updates progress."""
    workers_done = 0
    with tqdm(total=total_images, desc="Upscaling Frames") as pbar:
        while workers_done < num_workers:
            item = q_out.get()
            if item is None:
                workers_done += 1
                continue
            
            sr_image, output_path = item
            if sr_image:
                try:
                    sr_image.save(output_path)
                except Exception as e:
                    print(f"\nError saving {output_path}: {e}")
            
            pbar.update(1)

def main():
    # For AMD GPUs on Linux (ROCm), torch uses the 'cuda' alias.
    # For AMD GPUs on macOS, torch uses the 'mps' alias.
    if torch.cuda.is_available():
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices = [torch.device('mps')]
    else:
        devices = [torch.device('cpu')]

    num_workers = len(devices) if devices[0].type != 'cpu' else 0

    print(f"Detected {len(devices)} device(s):")
    if num_workers > 0:
        for device in devices:
            if device.type == 'cuda':
                print(f"  - GPU: {device} -> {torch.cuda.get_device_name(device)}")
            else: # MPS
                print(f"  - GPU: {device}")
        print(f"Will use {num_workers} GPU worker(s).")
    else:
        print("  - CPU: No supported GPU found. Using CPU.")
        print("\nWARNING: No GPU detected by PyTorch.")
        print("If you are on Linux with an AMD GPU, ensure you have ROCm installed and ")
        print("the ROCm version of PyTorch (check: pip list | grep torch).")
        print("\nPRO TIP: If your AMD GPU is present but not 'found', run the diagnostics")
        print("above to find each GPU's GFX version, then try one of:")
        print("    HSA_OVERRIDE_GFX_VERSION=10.3.5 HIP_VISIBLE_DEVICES=0 python upscale_frames.py  # iGPU example")
        print("    HSA_OVERRIDE_GFX_VERSION=10.3.0 HIP_VISIBLE_DEVICES=1 python upscale_frames.py  # dGPU example")
        print("    HSA_OVERRIDE_GFX_VERSION=11.0.3 HIP_VISIBLE_DEVICES=0 python upscale_frames.py  # Radeon 780M")
    print("---------------------------------------------------------------------------\n")

    # Determine which model will be used and build the output directory name
    # with a suffix so the user can tell at a glance which model was applied.
    anime_weights = 'weights/RealESRGAN_x4plus_anime_6B.pth'
    if os.path.exists(anime_weights) and RRDBNet:
        _model_key = 'RealESRGAN_x4plus_anime_6B'
    else:
        _model_key = 'RealESRGAN_x4plus'
    from upscaler.config import Config
    OUTPUT_DIR = f"{_OUTPUT_BASE}_{Config.model_output_suffix(_model_key)}"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all PNG images in the input directory
    images = sorted(glob(os.path.join(INPUT_DIR, '*.png')))
    total_frames = len(images)
    if not images:
        print(f"Error: No .png files found in the '{INPUT_DIR}' directory.")
        sys.exit(1)

    # Filter out images that have already been upscaled to allow for resuming.
    existing_files = {os.path.basename(f) for f in glob(os.path.join(OUTPUT_DIR, '*.png'))}
    images_to_process = [
        img for img in images if os.path.basename(img) not in existing_files
    ]
    skipped_frames = total_frames - len(images_to_process)

    print(f"Found {total_frames} total frames in '{INPUT_DIR}'.")
    if skipped_frames > 0:
        print(f"Skipping {skipped_frames} frames that already exist in '{OUTPUT_DIR}'.")

    if not images_to_process:
        print("All frames have already been upscaled. Nothing to do.")
        sys.exit(0)

    print(f"Processing {len(images_to_process)} new frames...")

    if num_workers > 0:
        # Multi-GPU / I/O optimized path
        q_in = Queue(maxsize=num_workers * 4)
        q_out = Queue(maxsize=num_workers * 4)

        loader = threading.Thread(target=loader_thread, args=(images_to_process, q_in, num_workers, OUTPUT_DIR))
        loader.start()

        processors = []
        for device in devices:
            worker = threading.Thread(target=processor_thread, args=(device, q_in, q_out))
            worker.start()
            processors.append(worker)

        saver = threading.Thread(target=saver_thread, args=(q_out, len(images_to_process), num_workers))
        saver.start()

        loader.join()
        for p in processors:
            p.join()
        saver.join()

    else:
        # Original single-threaded CPU path
        device = devices[0]
        model = RealESRGAN(device, scale=4)
        if os.path.exists(anime_weights) and RRDBNet:
            print(f"Using fast Anime 6B model: {anime_weights}")
            model.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model.model.to(device)
            model.load_weights(anime_weights, download=False)
        else:
            print("Using standard x4plus model (High Quality/Slower)")
            model.load_weights('weights/RealESRGAN_x4plus.pth', download=False)

        with torch.inference_mode():
            for path_to_image in tqdm(images_to_process, desc="Upscaling Frames"):
                try:
                    image = Image.open(path_to_image).convert('RGB')
                    sr_image = model.predict(image)
                    basename = os.path.basename(path_to_image)
                    output_path = os.path.join(OUTPUT_DIR, basename)
                    sr_image.save(output_path)
                except Exception as e:
                    print(f"\nError processing {path_to_image}: {e}")

    print(f"\nProcessing complete. Upscaled frames are in '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()