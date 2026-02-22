from typing import Tuple
import numpy as np
from PIL import Image
import hashlib
from numpy.typing import NDArray

class ExeSteganography:
    def __init__(self) -> None:
        self.end_marker: bytes = b"<<END_OF_EXE>>"
        
    def calculate_md5(self, file_path: str) -> str:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def _prepare_image(self, image_path: str) -> Tuple[NDArray, int]:
        img = Image.open(image_path)
        img_array = np.array(img)
        capacity = img_array.size
        return img_array, capacity
    
    def extract_exe(self, stego_image_path: str, output_exe_path: str) -> bool:
        try:
            # Read the stego image
            img_array = np.array(Image.open(stego_image_path))
            flat_img = img_array.flatten()
            
            # Extract LSBs
            extracted_bytes = bytearray()
            bits_buffer = ''
            
            for pixel in flat_img:
                bits_buffer += str(pixel & 1)
                if len(bits_buffer) == 8:
                    extracted_bytes.append(int(bits_buffer, 2))
                    bits_buffer = ''
                    
                    if len(extracted_bytes) >= len(self.end_marker):
                        if self.end_marker in extracted_bytes:
                            break
            
            # Process extracted data
            data = bytes(extracted_bytes)
            exe_data = data[:-len(self.end_marker)-32]
            embedded_md5 = data[-len(self.end_marker)-32:-len(self.end_marker)].decode()
            
            # Save and verify extracted executable
            with open(output_exe_path, 'wb') as f:
                f.write(exe_data)
            
            extracted_md5 = self.calculate_md5(output_exe_path)
            if extracted_md5 == embedded_md5:
                print("[+] Executable extracted successfully")
                print(f"[+] MD5 verified: {extracted_md5}")
                return True
            else:
                print("[!] Warning: MD5 verification failed")
                print(f"[!] Expected: {embedded_md5}")
                print(f"[!] Got: {extracted_md5}")
                return False
                
        except Exception as e:
            print(f"[!] Extraction failed: {str(e)}")
            return False

def main():
    stego = ExeSteganography()
    print("[*] Extracting njRAT from hacked_ptit.jpg...")
    stego.extract_exe("hacked_ptit.jpg", "extracted_ptit.exe")
    print("[*] Done! Check extracted_ptit.exe")

if __name__ == "__main__":
    main()
