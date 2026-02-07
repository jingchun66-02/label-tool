from PIL import Image, ImageDraw
import os
import subprocess
import sys
import json
import shutil

def create_test_image(path):
    # Create a 200x200 white image
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    # Draw a red circle
    draw.ellipse((50, 50, 150, 150), fill='red', outline='red')
    img.save(path)
    print(f"Created test image at {path}")

def run_labeler():
    # Clean output
    if os.path.exists('test_output'):
        shutil.rmtree('test_output')
        
    cmd = [
        sys.executable, '-m', 'labeler_tool.cli',
        '--input', 'test_image.jpg',
        '--prompts', 'circle',
        '--output', 'test_output',
        '--threshold', '0.05' # Low threshold for abstract shapes
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("Command failed!")
        return False
        
    if not os.path.exists('test_output/labels.json'):
        print("Output file not found!")
        return False
        
    with open('test_output/labels.json') as f:
        data = json.load(f)
        print("Result JSON:", json.dumps(data, indent=2))
        
    return True

if __name__ == '__main__':
    create_test_image('test_image.jpg')
    success = run_labeler()
    if success:
        print("Verification SUCCESS!")
    else:
        print("Verification FAILED!")
