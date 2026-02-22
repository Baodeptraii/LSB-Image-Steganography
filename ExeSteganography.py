from typing import Tuple
import numpy as np
from PIL import Image
import hashlib
from numpy.typing import NDArray

class ExeSteganography:
    """
    A class implementing steganography techniques to hide executable files within images.
    
    This class provides methods to embed executable files into carrier images and extract
    them later, using LSB (Least Significant Bit) steganography. It includes integrity
    verification through MD5 hashing.
    
    Attributes:
        end_marker (bytes): A marker used to identify the end of embedded data.
    """
    
    def __init__(self) -> None:
        """Initialize the ExeSteganography object with an end marker."""
        self.end_marker: bytes = b"<<END_OF_EXE>>"
        
    def calculate_md5(self, file_path: str) -> str:
        """
        Calculate the MD5 hash of a file.
        
        Args:
            file_path: Path to the file for which to calculate the MD5 hash.
            
        Returns:
            The hexadecimal representation of the MD5 hash.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If there are issues reading the file.
        """
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def _prepare_image(self, image_path: str) -> Tuple[NDArray, int]:
        """
        Prepare the carrier image for steganography operations.
        
        Args:
            image_path: Path to the carrier image.
            
        Returns:
            A tuple containing the numpy array of the image and its pixel capacity.
            
        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image format is not supported.
        """
        img = Image.open(image_path)
        img_array = np.array(img)
        capacity = img_array.size
        return img_array, capacity
    
    def embed_exe(self, image_path: str, exe_path: str, output_path: str) -> bool:
        """
        Embed an executable file into a carrier image.
        
        This method hides an executable file within an image using LSB steganography.
        It also embeds an MD5 hash for integrity verification.
        
        Args:
            image_path: Path to the carrier image.
            exe_path: Path to the executable file to hide.
            output_path: Path where the stego image will be saved.
            
        Returns:
            True if embedding was successful, False otherwise.
            
        Raises:
            ValueError: If the image is too small to hold the executable.
            FileNotFoundError: If either input file does not exist.
        """
        try:
            # Read the carrier image
            img_array, capacity = self._prepare_image(image_path)
            
            # Read the executable file
            with open(exe_path, 'rb') as exe_file:
                exe_data = exe_file.read()
            
            # Calculate original MD5 hash
            original_md5 = self.calculate_md5(exe_path)
            
            # Prepare the data to hide
            data_to_hide = exe_data + original_md5.encode() + self.end_marker
            bits = ''.join(format(byte, '08b') for byte in data_to_hide)
            
            if len(bits) > capacity:
                raise ValueError(
                    f"Image too small to hide {len(exe_data)/1024:.2f}KB executable"
                )
            
            # Flatten and modify image
            flat_img = img_array.flatten()
            for i, bit in enumerate(bits):
                flat_img[i] = (flat_img[i] & ~1) | int(bit)
            
            # Save modified image
            modified_img = flat_img.reshape(img_array.shape)
            Image.fromarray(modified_img).save(output_path, 'PNG')
            
            print(f"[+] Successfully embedded {len(exe_data)/1024:.2f}KB executable")
            print(f"[+] Original MD5: {original_md5}")
            return True
            
        except Exception as e:
            print(f"[!] Embedding failed: {str(e)}")
            return False
    
    def extract_exe(self, stego_image_path: str, output_exe_path: str) -> bool:
        """
        Extract an executable file from a stego image.
        
        This method extracts a previously hidden executable from an image and verifies
        its integrity using the embedded MD5 hash.
        
        Args:
            stego_image_path: Path to the stego image containing the hidden executable.
            output_exe_path: Path where the extracted executable will be saved.
            
        Returns:
            True if extraction and verification were successful, False otherwise.
            
        Raises:
            FileNotFoundError: If the stego image does not exist.
            ValueError: If the extracted data is corrupted or invalid.
        """
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

def main() -> None:
    """
    Main function demonstrating the usage of ExeSteganography class.
    """
    stego = ExeSteganography()
    
    print("[*] Image Steganography for Executable Files")
    print("[*] WARNING: For educational purposes only. Use in controlled environments.")
    
    # Embedding
    print("\n[*] Embedding executable...")
    if stego.embed_exe(image_path="../../ptit.jpg", 
                       exe_path="../../njRAT.exe", 
                       output_path="hacked_ptit.jpg"):
        # Extracting
        print("\n[*] Extracting executable...")
        stego.extract_exe(
            stego_image_path="hacked_ptit.jpg",
            output_exe_path="extracted_ptit.exe"
        )

if __name__ == "__main__":
    main()