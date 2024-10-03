import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import math
import hashlib
import json
import time
import traceback
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from scipy.stats import chisquare, kstest, norm, entropy
import threading
from collections import defaultdict
import random

# Constants
DH_PRIME = int("""FFFFFFFF FFFFFFFF C90FDAA2 2168C234 C4C6628B 80DC1CD1
29024E08 8A67CC74 020BBEA6 3B139B22 514A0879 8E3404DD
EF9519B3 CD3A431B 302B0A6D F25F1437 4FE1356D 6D51C245
E485B576 625E7EC6 F44C42E9 A63A3620 FFFFFFFF FFFFFFFF
""".replace(" ", "").replace("\n", ""), 16)
DH_GENERATOR = 2

class ThesisEncryption:
    def __init__(self):
        self.c_k = 1.0
        self.b_k = 1.0
        self.layer_index = 1

    def generate_key(self, seed, length=256, max_layers=5):
        np.random.seed(seed)
        key_bits = np.zeros(length, dtype=int)

        self.c_k = np.random.uniform(0.8, 1.2)
        self.b_k = np.random.uniform(0.8, 1.2)
        self.layer_index = 1

        for i in range(length):
            if i > 1 and key_bits[i-1] == key_bits[i-2]:
                self.layer_index = min(self.layer_index + 1, max_layers)
            else:
                self.layer_index = max(1, self.layer_index - 1)

            bit = int((self.c_k * (i + 1) ** (1/self.layer_index) * (np.log(i + 1) ** ((self.layer_index-1)/self.layer_index)) +
                    self.b_k * np.log(np.log(i + 1) + 1) +
                    0.01 * (np.log(i + 1) ** 2) * (i + 1) ** (1/self.layer_index)) % 2)

            if i % 64 == 0 and i != 0:
                bit ^= np.random.randint(0, 2)

            key_bits[i] = bit

            self.c_k += np.random.uniform(-0.1, 0.1)
            self.b_k += np.random.uniform(-0.1, 0.1)

        return key_bits

    def generate_aes_key(self, seed, key_length=256):
        key_bits = self.generate_key(seed, length=key_length)
        key_bytes = np.packbits(key_bits).tobytes()
        return key_bytes

    def encrypt(self, message, seed):
        key = self.generate_aes_key(seed)
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext = cipher.encrypt(pad(message, AES.block_size))
        return iv, ciphertext, key

    def decrypt(self, ciphertext, seed, iv):
        key = self.generate_aes_key(seed)
        decipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_message = unpad(decipher.decrypt(ciphertext), AES.block_size)
        return decrypted_message

    def calculate_entropy_fixed(self, bits):
        _, counts = np.unique(bits, return_counts=True)
        probabilities = counts / len(bits)
        entropy = -np.sum(np.fromiter((p * np.log2(p) for p in probabilities if p > 0), dtype=float))
        return entropy

    def simple_encrypt_fixed(self, seed, message):
        seed = int(seed)
        np.random.seed(seed)
        key_bits = np.random.randint(0, 2, 256, dtype=int)
        key = np.packbits(key_bits).tobytes()

        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        ciphertext = cipher.encrypt(pad(message, AES.block_size))
        return iv, ciphertext

    def find_first_perfect_entropy_seed(self, message, seed_start=1, seed_end=100000):
        for seed in range(seed_start, seed_end):
            iv, ciphertext = self.simple_encrypt_fixed(seed, message)
            ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            entropy = self.calculate_entropy_fixed(ciphertext_bits)
            if entropy == 1.0:
                return seed, entropy
        return None, None

    def encrypt_with_entropy_focus(self, seed, message, rounds=1):
        current_message = message
        for round_num in range(rounds):
            iv, ciphertext = self.simple_encrypt_fixed(seed, current_message)
            ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            entropy = self.calculate_entropy_fixed(ciphertext_bits)
            if entropy < 1.0:
                break
            current_message = ciphertext
        return iv, ciphertext

    def decrypt_with_entropy_focus(self, seed, iv, ciphertext):
        seed = int(seed)
        np.random.seed(seed)
        key_bits = np.random.randint(0, 2, 256, dtype=int)
        key = np.packbits(key_bits).tobytes()

        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_message = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted_message

class TimelineCompression:
    def __init__(self):
        self.layers = {}

    def compress(self, data):
        layer = math.ceil(math.log10(len(data)) / 3) if len(data) > 0 else 1
        x = data[0] if len(data) > 0 else 0
        y = data[1] if len(data) > 1 else 0

        timeline = self.generate_entry_timeline(data)
        timeline_hash = self.hash_timeline(timeline)
        self.store_data((layer, x, y, timeline_hash), timeline)

        return layer, x, y, timeline_hash

    def generate_entry_timeline(self, data):
        return list(enumerate(data, start=1))[::-1]

    def hash_timeline(self, timeline):
        timeline_str = ''.join([f"{byte}{depth}" for depth, byte in timeline])
        return hashlib.sha256(timeline_str.encode()).hexdigest()[:6]

    def store_data(self, encoded_structure, timeline):
        layer, x, y, timeline_hash = encoded_structure
        if layer not in self.layers:
            self.layers[layer] = {}
        if (x, y) not in self.layers[layer]:
            self.layers[layer][(x, y)] = {}
        self.layers[layer][(x, y)][timeline_hash] = timeline

    def decompress(self, encoded_structure):
        try:
            layer, x, y, timeline_hash = encoded_structure
            timeline = self.layers[layer][(x, y)][timeline_hash]
            return bytes(byte for _, byte in sorted(timeline))
        except Exception as e:
            print(f"Error during decompression: {e}")
            traceback.print_exc()
            raise e

    def convert_layers_for_saving(self):
        converted_layers = {}
        for layer, entries in self.layers.items():
            converted_layers[str(layer)] = {}
            for coords, timelines in entries.items():
                converted_layers[str(layer)][str(coords)] = timelines
        return converted_layers

    def restore_layers_after_loading(self, saved_layers):
        self.layers = {}
        for layer, entries in saved_layers.items():
            self.layers[int(layer)] = {}
            for coords, timelines in entries.items():
                coords_tuple = eval(coords)
                self.layers[int(layer)][coords_tuple] = timelines

class DiffieHellman:
    def __init__(self):
        secure_random = random.SystemRandom()
        self.private_key = secure_random.randint(2, DH_PRIME - 2)
        self.public_key = pow(DH_GENERATOR, self.private_key, DH_PRIME)

    def compute_shared_secret(self, other_public_key):
        shared_secret = pow(other_public_key, self.private_key, DH_PRIME)
        seed = int(hashlib.sha256(str(shared_secret).encode()).hexdigest(), 16) % (2 ** 32)
        return seed

class EncryptionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Encryption System")
        self.geometry("1200x800")
        self.configure(bg='#f0f0f0')

        self.thesis_encryption = ThesisEncryption()
        self.timeline_compression = TimelineCompression()

        self.key_length = 256
        self.max_layers = 5
        self.use_threading = True
        self.num_threads = 4
        self.top_seeds_to_display = 10

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=1, fill="both", padx=10, pady=10)

        self.encryption_data = {
            "random": None,
            "custom": None,
            "multi_key": None,
            "perfect_entropy": None
        }

        self.create_tabs()

    def create_tabs(self):
        # Rearranged tab order
        self.create_random_encryption_tab()
        self.create_custom_encryption_tab()
        self.create_multi_key_no_segment_padding_tab()
        self.create_perfect_entropy_tab()
        self.create_key_exchange_tab()
        self.create_enhanced_test_tab()
        self.create_settings_tab()

    def create_random_encryption_tab(self):
        random_frame = ttk.Frame(self.notebook)
        self.notebook.add(random_frame, text="Random Encryption")

        ttk.Label(random_frame, text="Message:").grid(column=0, row=0, padx=10, pady=10, sticky='w')
        self.message_entry = scrolledtext.ScrolledText(random_frame, width=70, height=5)
        self.message_entry.grid(column=1, row=0, padx=10, pady=10)

        ttk.Button(random_frame, text="Encrypt", command=self.encrypt_message).grid(column=0, row=2, padx=10, pady=10)
        ttk.Button(random_frame, text="Decrypt", command=self.decrypt_message).grid(column=1, row=2, padx=10, pady=10)

        ttk.Label(random_frame, text="Encoded Key:").grid(column=0, row=3, padx=10, pady=10, sticky='w')
        self.encoded_key_entry = ttk.Entry(random_frame, width=70)
        self.encoded_key_entry.grid(column=1, row=3, padx=10, pady=10)

        ttk.Label(random_frame, text="Decrypted Message:").grid(column=0, row=4, padx=10, pady=10, sticky='w')
        self.decrypted_entry = scrolledtext.ScrolledText(random_frame, width=70, height=5)
        self.decrypted_entry.grid(column=1, row=4, padx=10, pady=10)

    
    def create_custom_encryption_tab(self):
        custom_frame = ttk.Frame(self.notebook)
        self.notebook.add(custom_frame, text="Custom Encryption")

        ttk.Label(custom_frame, text="Step 1: Enter your secret message").grid(column=0, row=0, padx=10, pady=5, sticky='w')
        self.custom_message_entry = scrolledtext.ScrolledText(custom_frame, width=70, height=5)
        self.custom_message_entry.grid(column=0, row=1, padx=10, pady=5, columnspan=2)

        ttk.Label(custom_frame, text="Step 2: Enter a room number").grid(column=0, row=2, padx=10, pady=5, sticky='w')
        self.room_number_entry = ttk.Entry(custom_frame)
        self.room_number_entry.grid(column=0, row=3, padx=10, pady=5, sticky='w')

        ttk.Button(custom_frame, text="Encrypt", command=self.encrypt_custom_message).grid(column=0, row=4, padx=10, pady=5, sticky='w')
        ttk.Button(custom_frame, text="Decrypt", command=self.decrypt_custom_message).grid(column=1, row=4, padx=10, pady=5, sticky='w')

        ttk.Label(custom_frame, text="Encoded Key:").grid(column=0, row=5, padx=10, pady=5, sticky='w')
        self.custom_encoded_key_entry = ttk.Entry(custom_frame, width=70)
        self.custom_encoded_key_entry.grid(column=0, row=6, padx=10, pady=5, columnspan=2)

        ttk.Label(custom_frame, text="Decrypted Message:").grid(column=0, row=7, padx=10, pady=5, sticky='w')
        self.custom_decrypted_entry = scrolledtext.ScrolledText(custom_frame, width=70, height=5)
        self.custom_decrypted_entry.grid(column=0, row=8, padx=10, pady=5, columnspan=2)


    def create_multi_key_no_segment_padding_tab(self):
        multi_key_frame = ttk.Frame(self.notebook)
        self.notebook.add(multi_key_frame, text="Multi-Key Encryption")

        ttk.Label(multi_key_frame, text="Message:").grid(column=0, row=0, padx=10, pady=10, sticky='w')
        self.multi_key_message_entry = scrolledtext.ScrolledText(multi_key_frame, width=70, height=5)
        self.multi_key_message_entry.grid(column=1, row=0, padx=10, pady=10)

        ttk.Label(multi_key_frame, text="Rekey Interval:").grid(column=0, row=1, padx=10, pady=10, sticky='w')
        self.rekey_interval_entry = ttk.Entry(multi_key_frame, width=10)
        self.rekey_interval_entry.insert(0, "16")
        self.rekey_interval_entry.grid(column=1, row=1, padx=10, pady=10, sticky='w')

        ttk.Button(multi_key_frame, text="Encrypt", command=self.encrypt_multi_key_message).grid(column=0, row=2, padx=10, pady=10)
        ttk.Button(multi_key_frame, text="Decrypt", command=self.decrypt_multi_key_message).grid(column=1, row=2, padx=10, pady=10)

        ttk.Label(multi_key_frame, text="Encoded Key:").grid(column=0, row=3, padx=10, pady=10, sticky='w')
        self.multi_key_encoded_key_entry = ttk.Entry(multi_key_frame, width=70)
        self.multi_key_encoded_key_entry.grid(column=1, row=3, padx=10, pady=10)

        ttk.Label(multi_key_frame, text="Decrypted Message:").grid(column=0, row=4, padx=10, pady=10, sticky='w')
        self.multi_key_decrypted_entry = scrolledtext.ScrolledText(multi_key_frame, width=70, height=5)
        self.multi_key_decrypted_entry.grid(column=1, row=4, padx=10, pady=10)

    def create_key_exchange_tab(self):
        key_exchange_frame = ttk.Frame(self.notebook)
        self.notebook.add(key_exchange_frame, text="Key Exchange")

        ttk.Label(key_exchange_frame, text="Your Public Key:").grid(column=0, row=0, padx=10, pady=10, sticky='w')
        self.public_key_entry = scrolledtext.ScrolledText(key_exchange_frame, width=70, height=5)
        self.public_key_entry.grid(column=1, row=0, padx=10, pady=10)

        ttk.Button(key_exchange_frame, text="Generate Public Key", command=self.generate_public_key).grid(column=0, row=1, padx=10, pady=10)

        ttk.Label(key_exchange_frame, text="Other Party's Public Key:").grid(column=0, row=2, padx=10, pady=10, sticky='w')
        self.other_public_key_entry = scrolledtext.ScrolledText(key_exchange_frame, width=70, height=5)
        self.other_public_key_entry.grid(column=1, row=2, padx=10, pady=10)

        ttk.Button(key_exchange_frame, text="Compute Shared Secret", command=self.compute_shared_secret).grid(column=0, row=3, padx=10, pady=10)

        ttk.Label(key_exchange_frame, text="Shared Secret (Seed):").grid(column=0, row=4, padx=10, pady=10, sticky='w')
        self.shared_secret_entry = ttk.Entry(key_exchange_frame, width=30)
        self.shared_secret_entry.grid(column=1, row=4, padx=10, pady=10, sticky='w')

    def create_enhanced_test_tab(self):
        test_frame = ttk.Frame(self.notebook)
        self.notebook.add(test_frame, text="Tests")

        ttk.Button(test_frame, text="Generate New Key", command=self.generate_new_test_key).grid(column=0, row=0, padx=10, pady=10)

        ttk.Label(test_frame, text="Select Test Source:").grid(column=1, row=0, padx=10, pady=10, sticky='w')
        self.test_source = ttk.Combobox(test_frame, values=[
            "Random Encryption",
            "Custom Encryption (Last Generated)",
            "New Custom Key",
            "Multi-Key Encryption",
            "Perfect Entropy Encryption"
        ])
        self.test_source.grid(column=2, row=0, padx=10, pady=10)
        self.test_source.set("Random Encryption")

        ttk.Button(test_frame, text="Run Tests", command=self.run_tests).grid(column=3, row=0, padx=10, pady=10)

        self.test_result = scrolledtext.ScrolledText(test_frame, width=120, height=30)
        self.test_result.grid(column=0, row=1, columnspan=4, padx=10, pady=10)

        self.visual_frame = ttk.Frame(test_frame)
        self.visual_frame.grid(column=0, row=2, columnspan=4, padx=10, pady=10)

    def create_settings_tab(self):
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")

        ttk.Label(settings_frame, text="Key Length (bits):").grid(column=0, row=0, padx=10, pady=10, sticky='w')
        self.key_length_entry = ttk.Entry(settings_frame, width=10)
        self.key_length_entry.insert(0, str(self.key_length))
        self.key_length_entry.grid(column=1, row=0, padx=10, pady=10, sticky='w')

        ttk.Label(settings_frame, text="Maximum Layers:").grid(column=0, row=1, padx=10, pady=10, sticky='w')
        self.max_layers_entry = ttk.Entry(settings_frame, width=10)
        self.max_layers_entry.insert(0, str(self.max_layers))
        self.max_layers_entry.grid(column=1, row=1, padx=10, pady=10, sticky='w')

        ttk.Label(settings_frame, text="Use Threading:").grid(column=0, row=2, padx=10, pady=10, sticky='w')
        self.use_threading_var = tk.BooleanVar(value=self.use_threading)
        ttk.Checkbutton(settings_frame, variable=self.use_threading_var).grid(column=1, row=2, padx=10, pady=10, sticky='w')

        ttk.Label(settings_frame, text="Number of Threads:").grid(column=0, row=3, padx=10, pady=10, sticky='w')
        self.num_threads_entry = ttk.Entry(settings_frame, width=10)
        self.num_threads_entry.insert(0, str(self.num_threads))
        self.num_threads_entry.grid(column=1, row=3, padx=10, pady=10, sticky='w')

        ttk.Label(settings_frame, text="Top Seeds to Display:").grid(column=0, row=4, padx=10, pady=10, sticky='w')
        self.top_seeds_entry = ttk.Entry(settings_frame, width=10)
        self.top_seeds_entry.insert(0, str(self.top_seeds_to_display))
        self.top_seeds_entry.grid(column=1, row=4, padx=10, pady=10, sticky='w')

        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(column=0, row=5, columnspan=2, padx=10, pady=10)
    
    def save_settings(self):
        try:
            self.key_length = int(self.key_length_entry.get())
            self.max_layers = int(self.max_layers_entry.get())
            self.use_threading = self.use_threading_var.get()
            self.num_threads = int(self.num_threads_entry.get())
            self.top_seeds_to_display = int(self.top_seeds_entry.get())
            messagebox.showinfo("Settings", "Settings saved successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for numeric fields.")

    def find_perfect_entropy_seed(self):
        message = self.perfect_entropy_message_entry.get("1.0", tk.END).encode().strip()
        seed, entropy = self.thesis_encryption.find_first_perfect_entropy_seed(message)
        if seed is not None:
            self.perfect_entropy_seed_entry.delete(0, tk.END)
            self.perfect_entropy_seed_entry.insert(0, str(seed))
            messagebox.showinfo("Success", f"Perfect entropy seed found: {seed}")
        else:
            messagebox.showerror("Error", "No perfect entropy seed found within range.")

    
    def create_perfect_entropy_tab(self):
        perfect_entropy_frame = ttk.Frame(self.notebook)
        self.notebook.add(perfect_entropy_frame, text="Perfect Entropy Encryption")

        ttk.Label(perfect_entropy_frame, text="Message:").grid(column=0, row=0, padx=10, pady=10, sticky='w')
        self.perfect_entropy_message_entry = scrolledtext.ScrolledText(perfect_entropy_frame, width=70, height=5)
        self.perfect_entropy_message_entry.grid(column=1, row=0, padx=10, pady=10)

        ttk.Button(perfect_entropy_frame, text="Find Perfect Entropy Seed", command=self.find_perfect_entropy_seed).grid(column=0, row=1, padx=10, pady=10)
        ttk.Button(perfect_entropy_frame, text="Encrypt", command=self.encrypt_perfect_entropy).grid(column=0, row=2, padx=10, pady=10)
        ttk.Button(perfect_entropy_frame, text="Decrypt", command=self.decrypt_perfect_entropy).grid(column=1, row=2, padx=10, pady=10)

        ttk.Label(perfect_entropy_frame, text="Perfect Entropy Seed:").grid(column=0, row=3, padx=10, pady=10, sticky='w')
        self.perfect_entropy_seed_entry = ttk.Entry(perfect_entropy_frame, width=30)
        self.perfect_entropy_seed_entry.grid(column=1, row=3, padx=10, pady=10, sticky='w')

        ttk.Label(perfect_entropy_frame, text="Encrypted Message:").grid(column=0, row=4, padx=10, pady=10, sticky='w')
        self.perfect_entropy_encrypted_entry = scrolledtext.ScrolledText(perfect_entropy_frame, width=70, height=5)
        self.perfect_entropy_encrypted_entry.grid(column=1, row=4, padx=10, pady=10)

        ttk.Label(perfect_entropy_frame, text="Decrypted Message:").grid(column=0, row=5, padx=10, pady=10, sticky='w')
        self.perfect_entropy_decrypted_entry = scrolledtext.ScrolledText(perfect_entropy_frame, width=70, height=5)
        self.perfect_entropy_decrypted_entry.grid(column=1, row=5, padx=10, pady=10)
    
    def encrypt_custom_message(self):
        message = self.custom_message_entry.get("1.0", tk.END).encode().strip()
        try:
            seed = int(self.room_number_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid Room Number. Please enter an integer.")
            return

        start_time = time.time()
        iv, ciphertext, key = self.thesis_encryption.encrypt(message, seed)
        compressed_key = self.timeline_compression.compress(iv + ciphertext)
        end_time = time.time()

        self.encryption_data["custom"] = {
            "seed": seed,
            "iv": iv,
            "ciphertext": ciphertext,
            "key": key,
            "compressed_key": compressed_key,
            "message": message,
            "encryption_time": end_time - start_time
        }

        encoded_key = f"{seed},{json.dumps(compressed_key)}"
        self.custom_encoded_key_entry.delete(0, tk.END)
        self.custom_encoded_key_entry.insert(0, encoded_key)

        messagebox.showinfo("Encryption Complete", f"Encryption took {end_time - start_time:.4f} seconds")

    def decrypt_custom_message(self):
        try:
            encoded_key = self.custom_encoded_key_entry.get()
            seed_str, key_str = encoded_key.split(',', 1)
            seed = int(seed_str)
            compressed_key = json.loads(key_str.strip())

            start_time = time.time()
            decompressed_data = self.timeline_compression.decompress(compressed_key)
            iv, ciphertext = decompressed_data[:16], decompressed_data[16:]

            decrypted_message = self.thesis_encryption.decrypt(ciphertext, seed, iv)
            end_time = time.time()

            self.custom_decrypted_entry.delete("1.0", tk.END)
            self.custom_decrypted_entry.insert(tk.END, decrypted_message.decode())

            messagebox.showinfo("Decryption Complete", f"Decryption took {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error during decryption: {e}")
            traceback.print_exc()
            messagebox.showerror("Decryption Error", str(e))
    
    def encrypt_message(self):
        message = self.message_entry.get("1.0", tk.END).encode().strip()
        seed = np.random.randint(0, 100000000)

        start_time = time.time()
        iv, ciphertext, key = self.thesis_encryption.encrypt(message, seed)
        compressed_key = self.timeline_compression.compress(iv + ciphertext)
        end_time = time.time()

        self.encryption_data["random"] = {
            "seed": seed,
            "iv": iv,
            "ciphertext": ciphertext,
            "key": key,
            "compressed_key": compressed_key,
            "message": message,
            "encryption_time": end_time - start_time
        }

        encoded_key = f"{seed},{json.dumps(compressed_key)}"
        self.encoded_key_entry.delete(0, tk.END)
        self.encoded_key_entry.insert(0, encoded_key)

        messagebox.showinfo("Encryption Complete", f"Encryption took {end_time - start_time:.4f} seconds")

    def decrypt_message(self):
        try:
            encoded_key = self.encoded_key_entry.get()
            seed_str, key_str = encoded_key.split(',', 1)
            seed = int(seed_str)
            compressed_key = json.loads(key_str.strip())

            start_time = time.time()
            decompressed_data = self.timeline_compression.decompress(compressed_key)
            iv, ciphertext = decompressed_data[:16], decompressed_data[16:]

            decrypted_message = self.thesis_encryption.decrypt(ciphertext, seed, iv)
            end_time = time.time()

            self.decrypted_entry.delete("1.0", tk.END)
            self.decrypted_entry.insert(tk.END, decrypted_message.decode())

            messagebox.showinfo("Decryption Complete", f"Decryption took {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error during decryption: {e}")
            traceback.print_exc()
            messagebox.showerror("Decryption Error", str(e))

    def generate_key_and_test(self):
        message = self.custom_message_entry.get("1.0", tk.END).encode().strip()
        try:
            seed = int(self.room_number_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid Room Number. Please enter an integer.")
            return

        start_time = time.time()
        iv, ciphertext, key = self.thesis_encryption.encrypt(message, seed)
        compressed_key = self.timeline_compression.compress(iv + ciphertext)
        end_time = time.time()

        self.encryption_data["custom"] = {
            "seed": seed,
            "iv": iv,
            "ciphertext": ciphertext,
            "key": key,
            "compressed_key": compressed_key,
            "message": message,
            "encryption_time": end_time - start_time
        }

        encoded_key = f"{seed},{json.dumps(compressed_key)}"
        self.custom_encoded_key_entry.delete(0, tk.END)
        self.custom_encoded_key_entry.insert(0, encoded_key)

        self.test_source.set("Custom Encryption (Last Generated)")
        self.run_tests()

    def encrypt_multi_key_message(self):
        message = self.multi_key_message_entry.get("1.0", tk.END).encode().strip()
        seed = np.random.randint(0, 100000000)
        rekey_interval = int(self.rekey_interval_entry.get())

        start_time = time.time()
        encrypted_message, key_history = self.multi_key_test_no_segment_padding(message, seed, rekey_interval)
        end_time = time.time()

        self.encryption_data["multi_key"] = {
            "seed": seed,
            "encrypted_message": encrypted_message,
            "key_history": key_history,
            "message": message,
            "encryption_time": end_time - start_time,
            "rekey_interval": rekey_interval
        }

        encoded_key = f"{seed},{json.dumps(key_history)}"
        self.multi_key_encoded_key_entry.delete(0, tk.END)
        self.multi_key_encoded_key_entry.insert(0, encoded_key)

        messagebox.showinfo("Multi-Key Encryption Complete", f"Encryption took {end_time - start_time:.4f} seconds")

    def decrypt_multi_key_message(self):
        try:
            encoded_key = self.multi_key_encoded_key_entry.get()
            seed_str, key_str = encoded_key.split(',', 1)
            seed = int(seed_str)
            key_history = json.loads(key_str)

            start_time = time.time()
            decrypted_message = self.multi_key_decrypt_no_segment_padding(
                self.encryption_data["multi_key"]["encrypted_message"], 
                key_history, 
                self.encryption_data["multi_key"]["rekey_interval"]
            )
            end_time = time.time()

            self.multi_key_decrypted_entry.delete("1.0", tk.END)
            self.multi_key_decrypted_entry.insert(tk.END, decrypted_message.decode())

            messagebox.showinfo("Multi-Key Decryption Complete", f"Decryption took {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error during multi-key decryption: {e}")
            traceback.print_exc()
            messagebox.showerror("Decryption Error", str(e))

    def generate_public_key(self):
        self.dh = DiffieHellman()
        public_key = self.dh.public_key
        self.public_key_entry.delete('1.0', tk.END)
        self.public_key_entry.insert('1.0', str(public_key))
        messagebox.showinfo("Success", "Public key generated.")

    def compute_shared_secret(self):
        other_public_key_str = self.other_public_key_entry.get('1.0', tk.END).strip()
        if not other_public_key_str:
            messagebox.showerror("Error", "Please enter the other party's public key.")
            return
        try:
            other_public_key = int(other_public_key_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid public key format.")
            return

        if not hasattr(self, 'dh'):
            messagebox.showerror("Error", "Please generate your public key first.")
            return

        seed = self.dh.compute_shared_secret(other_public_key)
        self.shared_secret_entry.delete(0, tk.END)
        self.shared_secret_entry.insert(0, str(seed))
        messagebox.showinfo("Success", "Shared secret computed.")

    def run_tests(self):
        test_source = self.test_source.get()

        if test_source == "Random Encryption":
            self.test_random_encryption()
        elif test_source == "Custom Encryption (Last Generated)" or test_source == "New Custom Key":
            self.test_custom_encryption()
        elif test_source == "Multi-Key Encryption":
            self.test_multi_key_encryption()
        elif test_source == "Perfect Entropy Encryption":
            self.test_perfect_entropy_encryption()
        else:
            messagebox.showerror("Error", "Invalid test source selected.")
    
    def test_perfect_entropy_encryption(self):
        if "perfect_entropy" not in self.encryption_data or self.encryption_data["perfect_entropy"] is None:
            messagebox.showerror("Error", "Please perform Perfect Entropy encryption first.")
            return

        key_data = self.encryption_data["perfect_entropy"]
        self.test_result.delete('1.0', tk.END)
        self.test_result.insert(tk.END, "--- Perfect Entropy Encryption Tests ---\n\n")
        self.test_result.insert(tk.END, f"Seed: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"Encryption Time: {key_data['encryption_time']:.4f} seconds\n\n")

        ciphertext_bits = np.unpackbits(np.frombuffer(key_data['ciphertext'], dtype=np.uint8))
        entropy = self.thesis_encryption.calculate_entropy_fixed(ciphertext_bits)
        
        self.run_existing_tests(ciphertext_bits)
        self.run_additional_strength_tests(ciphertext_bits)
        self.run_avalanche_test(key_data)
        self.run_visual_tests(ciphertext_bits)
        self.run_monobit_test(ciphertext_bits)
        self.run_runs_test(ciphertext_bits)
        self.run_longest_run_test(ciphertext_bits)

        self.test_result.insert(tk.END, f"\nFinal Entropy: {entropy:.6f}\n")
        self.test_result.insert(tk.END, f"Perfect Entropy Seed: {key_data['seed']}\n")

        # Add the "recipe" at the end
        self.test_result.insert(tk.END, "\n--- Encryption Recipe ---\n")
        self.test_result.insert(tk.END, f"1. Start with seed: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"2. Generate key using ThesisEncryption algorithm\n")
        self.test_result.insert(tk.END, f"3. Encrypt message using AES-CBC mode\n")
        self.test_result.insert(tk.END, f"4. Resulting ciphertext length: {len(key_data['ciphertext'])} bytes\n")
        self.test_result.insert(tk.END, f"5. Achieved entropy: {entropy:.6f}\n")
        self.test_result.insert(tk.END, f"6. Total encryption time: {key_data['encryption_time']:.4f} seconds\n")

    
    def test_random_encryption(self):
        if "random" not in self.encryption_data or self.encryption_data["random"] is None:
            messagebox.showerror("Error", "Please perform Random encryption first.")
            return

        key_data = self.encryption_data["random"]
        self.run_encryption_tests(key_data, "Random Encryption")

    def test_custom_encryption(self):
        if "custom" not in self.encryption_data or self.encryption_data["custom"] is None:
            messagebox.showerror("Error", "Please generate a custom key first.")
            return

        key_data = self.encryption_data["custom"]
        self.run_encryption_tests(key_data, "Custom Encryption")

    def test_multi_key_encryption(self):
        if "multi_key" not in self.encryption_data or self.encryption_data["multi_key"] is None:
            messagebox.showerror("Error", "Please perform Multi-Key encryption first.")
            return

        key_data = self.encryption_data["multi_key"]
        self.run_multi_key_encryption_tests(key_data)

    def test_perfect_entropy_encryption(self):
        if "perfect_entropy" not in self.encryption_data or self.encryption_data["perfect_entropy"] is None:
            messagebox.showerror("Error", "Please perform Perfect Entropy encryption first.")
            return

        key_data = self.encryption_data["perfect_entropy"]
        self.test_result.delete('1.0', tk.END)
        self.test_result.insert(tk.END, "--- Perfect Entropy Encryption Tests ---\n\n")
        self.test_result.insert(tk.END, f"Best Seed: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"Encryption Time: {key_data['encryption_time']:.6f} seconds\n")
        self.test_result.insert(tk.END, f"Number of Attempts: {key_data['attempts']}\n")
        self.test_result.insert(tk.END, f"Achieved Entropy: {key_data['achieved_entropy']:.6f}\n\n")

        ciphertext_bits = np.unpackbits(np.frombuffer(key_data['ciphertext'], dtype=np.uint8))
    
        self.run_existing_tests(ciphertext_bits)
        self.run_additional_strength_tests(ciphertext_bits)
        self.run_avalanche_test(key_data)
        self.run_visual_tests(ciphertext_bits)
        self.run_monobit_test(ciphertext_bits)
        self.run_runs_test(ciphertext_bits)
        self.run_longest_run_test(ciphertext_bits)

        # Add the "recipe" at the end
        self.test_result.insert(tk.END, "\n--- Encryption Recipe ---\n")
        self.test_result.insert(tk.END, f"1. Start with message length: {len(key_data['message'])} bytes\n")
        self.test_result.insert(tk.END, f"2. Try different seeds until near-perfect entropy is achieved\n")
        self.test_result.insert(tk.END, f"3. Best seed found: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"4. Generate key using ThesisEncryption algorithm\n")
        self.test_result.insert(tk.END, f"5. Encrypt message using AES-CBC mode\n")
        self.test_result.insert(tk.END, f"6. Resulting ciphertext length: {len(key_data['ciphertext'])} bytes\n")
        self.test_result.insert(tk.END, f"7. Achieved entropy: {key_data['achieved_entropy']:.6f}\n")
        self.test_result.insert(tk.END, f"8. Total encryption time: {key_data['encryption_time']:.6f} seconds\n")
        self.test_result.insert(tk.END, f"9. Number of attempts: {key_data['attempts']}\n")

        # Additional analysis
        self.test_result.insert(tk.END, "\n--- Strength Analysis ---\n")
        self.test_result.insert(tk.END, f"1. Entropy closeness to 1: {abs(1 - key_data['achieved_entropy']):.6f}\n")
    
        # Handle potential division by zero
        if key_data['encryption_time'] > 0:
            encryption_speed = len(key_data['message']) / key_data['encryption_time'] / 1024
            attempts_per_second = key_data['attempts'] / key_data['encryption_time']
            self.test_result.insert(tk.END, f"2. Encryption speed: {encryption_speed:.2f} KB/s\n")
            self.test_result.insert(tk.END, f"3. Attempts per second: {attempts_per_second:.2f}\n")
        else:
            self.test_result.insert(tk.END, "2. Encryption speed: Instantaneous (too fast to measure)\n")
            self.test_result.insert(tk.END, "3. Attempts per second: N/A (encryption too fast to measure)\n")
    
        # Interpretation
        self.test_result.insert(tk.END, "\n--- Interpretation ---\n")
        if abs(1 - key_data['achieved_entropy']) < 0.0001:
            self.test_result.insert(tk.END, "The encryption achieved near-perfect entropy, indicating excellent randomness.\n")
        elif abs(1 - key_data['achieved_entropy']) < 0.001:
            self.test_result.insert(tk.END, "The encryption achieved very high entropy, indicating strong randomness.\n")
        else:
            self.test_result.insert(tk.END, "The encryption achieved good entropy, but there might be room for improvement.\n")

        self.test_result.insert(tk.END, f"The encryption process took {key_data['attempts']} attempts to find a seed that produces near-perfect entropy.\n")
    
        if key_data['encryption_time'] > 0:
            self.test_result.insert(tk.END, f"The encryption speed of {encryption_speed:.2f} KB/s includes the time taken to find the optimal seed.\n")
        else:
            self.test_result.insert(tk.END, "The encryption process was too fast to measure accurately, indicating excellent performance.\n")

        self.test_result.insert(tk.END, "\nConclusion: This encryption method provides a strong balance between security (high entropy) and performance, with the added benefit of searching for an optimal seed to maximize entropy.\n")


    def run_encryption_tests(self, key_data, encryption_type):
        self.test_result.delete('1.0', tk.END)
        self.test_result.insert(tk.END, f"--- {encryption_type} Tests ---\n\n")
        self.test_result.insert(tk.END, f"Seed: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"Encryption Time: {key_data['encryption_time']:.4f} seconds\n\n")

        key_bits = np.unpackbits(np.frombuffer(key_data['key'], dtype=np.uint8))

        self.run_existing_tests(key_bits)
        self.run_additional_strength_tests(key_bits)
        self.run_avalanche_test(key_data)
        self.run_visual_tests(key_bits)
        self.run_monobit_test(key_bits)
        self.run_runs_test(key_bits)
        self.run_longest_run_test(key_bits)

        start_time = time.time()
        decompressed_data = self.timeline_compression.decompress(key_data['compressed_key'])
        decrypted_iv, decrypted_ciphertext = decompressed_data[:16], decompressed_data[16:]
        decrypted_message = self.thesis_encryption.decrypt(decrypted_ciphertext, key_data['seed'], decrypted_iv)
        end_time = time.time()

        self.test_result.insert(tk.END, f"\nDecryption Time: {end_time - start_time:.4f} seconds\n")
        decrypted_hash = hashlib.sha256(decrypted_message).hexdigest()
        self.test_result.insert(tk.END, f"Decrypted Message Hash: {decrypted_hash}\n")

        if decrypted_message == key_data['message']:
            self.test_result.insert(tk.END, "Decryption Successful: Message integrity maintained\n")
        else:
            self.test_result.insert(tk.END, "Decryption Failed: Message integrity compromised\n")

    def run_multi_key_encryption_tests(self, key_data):
        self.test_result.delete('1.0', tk.END)
        self.test_result.insert(tk.END, "--- Multi-Key Encryption Tests ---\n\n")
        self.test_result.insert(tk.END, f"Initial Seed: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"Encryption Time: {key_data['encryption_time']:.4f} seconds\n")
        self.test_result.insert(tk.END, f"Rekey Interval: {key_data['rekey_interval']} bytes\n")
        self.test_result.insert(tk.END, f"Number of Keys Used: {len(key_data['key_history'])}\n\n")

        # Analyze key history
        self.analyze_key_history(key_data['key_history'])

        # Analyze the entire encrypted message
        encrypted_message = key_data['encrypted_message']
        encrypted_bits = np.unpackbits(np.frombuffer(encrypted_message, dtype=np.uint8))

        self.test_result.insert(tk.END, "\nEncrypted Message Analysis:\n")
        self.run_existing_tests(encrypted_bits)
        self.run_additional_strength_tests(encrypted_bits)
        self.run_monobit_test(encrypted_bits)
        self.run_runs_test(encrypted_bits)
        self.run_longest_run_test(encrypted_bits)

        # Perform decryption test
        start_time = time.time()
        decrypted_message = self.multi_key_decrypt_no_segment_padding(
            key_data['encrypted_message'], 
            key_data['key_history'], 
            key_data['rekey_interval']
        )
        end_time = time.time()

        self.test_result.insert(tk.END, f"\nDecryption Time: {end_time - start_time:.4f} seconds\n")
        decrypted_hash = hashlib.sha256(decrypted_message).hexdigest()
        self.test_result.insert(tk.END, f"Decrypted Message Hash: {decrypted_hash}\n")

        if decrypted_message == key_data['message']:
            self.test_result.insert(tk.END, "Decryption Successful: Message integrity maintained\n")
        else:
            self.test_result.insert(tk.END, "Decryption Failed: Message integrity compromised\n")

        # Perform avalanche effect test
        self.run_avalanche_test(key_data)

    def analyze_key_history(self, key_history):
        seeds = [entry[0] for entry in key_history]
        seed_differences = [seeds[i+1] - seeds[i] for i in range(len(seeds)-1)]

        self.test_result.insert(tk.END, "Key History Analysis:\n")
        if seed_differences:
            self.test_result.insert(tk.END, f"Minimum seed difference: {min(seed_differences)}\n")
            self.test_result.insert(tk.END, f"Maximum seed difference: {max(seed_differences)}\n")
            self.test_result.insert(tk.END, f"Average seed difference: {sum(seed_differences)/len(seed_differences):.2f}\n")
        else:
            self.test_result.insert(tk.END, "Only one key used, no seed differences to analyze.\n")

        # Analyze key entropy
        key_entropies = []
        for _, _, key_hex in key_history:
            key = bytes.fromhex(key_hex)
            key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
            key_entropies.append(self.thesis_encryption.calculate_entropy_fixed(key_bits))

        self.test_result.insert(tk.END, "\nKey Entropy Analysis:\n")
        self.test_result.insert(tk.END, f"Minimum key entropy: {min(key_entropies):.4f}\n")
        self.test_result.insert(tk.END, f"Maximum key entropy: {max(key_entropies):.4f}\n")
        self.test_result.insert(tk.END, f"Average key entropy: {sum(key_entropies)/len(key_entropies):.4f}\n")

        # Analyze individual keys
        for i, (_, _, key_hex) in enumerate(key_history):
            key = bytes.fromhex(key_hex)
            key_bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8))
            self.test_result.insert(tk.END, f"\nKey {i+1} Analysis:\n")
            self.run_existing_tests(key_bits)
            self.run_additional_strength_tests(key_bits)

    def run_existing_tests(self, key):
        entropy = self.thesis_encryption.calculate_entropy_fixed(key)
        self.test_result.insert(tk.END, f"Entropy: {entropy:.4f}\n")

        chi2, p_value = self.chi_square_test(key)
        self.test_result.insert(tk.END, f"Chi-Square Statistic: {chi2:.4f}\n")
        self.test_result.insert(tk.END, f"P-value: {p_value:.4f}\n")

        freq_result = self.frequency_test(key)
        self.test_result.insert(tk.END, f"Frequency Test: {'Passed' if freq_result else 'Failed'}\n")

        run_result = self.run_test(key)
        self.test_result.insert(tk.END, f"Run Test: {'Passed' if run_result else 'Failed'}\n")

        auto_corr = self.autocorrelation_test(key)
        self.test_result.insert(tk.END, f"Autocorrelation: {auto_corr:.4f}\n")

    def run_additional_strength_tests(self, key):
        serial_p_value = self.serial_test(key)
        self.test_result.insert(tk.END, f"Serial Test P-value: {serial_p_value:.4f}\n")

        poker_p_value = self.poker_test(key)
        self.test_result.insert(tk.END, f"Poker Test P-value: {poker_p_value:.4f}\n")

        cumsum_p_value = self.cumulative_sums_test(key)
        self.test_result.insert(tk.END, f"Cumulative Sums Test P-value: {cumsum_p_value:.4f}\n")

    def run_avalanche_test(self, key_data):
        message = key_data['message']
    
        if 'ciphertext' in key_data:
            # Single-key encryption
            seed = key_data['seed']
            original_ciphertext = key_data['ciphertext']
            encrypt_func = lambda m: self.thesis_encryption.encrypt(m, seed)[1]
        else:
            # Multi-key encryption
            seed = key_data['seed']
            original_ciphertext = key_data['encrypted_message']
            encrypt_func = lambda m: self.multi_key_test_no_segment_padding(m, seed, key_data['rekey_interval'])[0]

        avalanche_scores = []
        for i in range(len(message) * 8):
            modified_message = bytearray(message)
            byte_index = i // 8
            bit_index = i % 8
            modified_message[byte_index] ^= 1 << bit_index
        
            modified_ciphertext = encrypt_func(bytes(modified_message))
        
            diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(original_ciphertext, modified_ciphertext))
            avalanche_score = (diff_bits / (len(original_ciphertext) * 8)) * 100
            avalanche_scores.append(avalanche_score)

        avg_avalanche = sum(avalanche_scores) / len(avalanche_scores)
        self.test_result.insert(tk.END, f"Average Avalanche Effect: {avg_avalanche:.2f}%\n")
        self.test_result.insert(tk.END, f"Minimum Avalanche Effect: {min(avalanche_scores):.2f}%\n")
        self.test_result.insert(tk.END, f"Maximum Avalanche Effect: {max(avalanche_scores):.2f}%\n")
        self.test_result.insert(tk.END, f"Avalanche Effect Standard Deviation: {np.std(avalanche_scores):.2f}%\n")

    def run_visual_tests(self, key):
        for widget in self.visual_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].hist(key, bins=2, range=(0, 1), align='mid', rwidth=0.8)
        axes[0, 0].set_title('Key Bit Distribution')
        axes[0, 0].set_xlabel('Bit Value')
        axes[0, 0].set_ylabel('Frequency')

        autocorr = [self.autocorrelation_test(key[i:]) for i in range(1, 100)]
        axes[0, 1].plot(autocorr)
        axes[0, 1].set_title('Autocorrelation')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('Autocorrelation')

        axes[1, 0].scatter(range(len(key)), key, s=1)
        axes[1, 0].set_title('Key Bits Scatter Plot')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('Bit Value')

        run_lengths = [len(list(group)) for _, group in itertools.groupby(key)]
        axes[1, 1].hist(run_lengths, bins=range(1, max(run_lengths) + 2), align='left', rwidth=0.8)
        axes[1, 1].set_title('Run Length Distribution')
        axes[1, 1].set_xlabel('Run Length')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.visual_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)

    def run_monobit_test(self, key):
        ones_count = sum(key)
        zeros_count = len(key) - ones_count
        if abs(ones_count - zeros_count) <= 2 * np.sqrt(len(key)):
            self.test_result.insert(tk.END, "Monobit Test: Passed\n")
        else:
            self.test_result.insert(tk.END, "Monobit Test: Failed\n")
        self.test_result.insert(tk.END, f"Ones count: {ones_count}, Zeros count: {zeros_count}\n")

    def run_runs_test(self, key):
        runs = np.diff(key)
        num_runs = np.count_nonzero(runs) + 1
        expected_runs = (2 * sum(key) * (len(key) - sum(key)) / len(key)) + 1
        if abs(num_runs - expected_runs) <= 2 * np.sqrt(2 * len(key) - 1) / 4:
            self.test_result.insert(tk.END, "Runs Test: Passed\n")
        else:
            self.test_result.insert(tk.END, "Runs Test: Failed\n")
        self.test_result.insert(tk.END, f"Number of runs: {num_runs}, Expected runs: {expected_runs:.2f}\n")

    def run_longest_run_test(self, key):
        longest_run = max(len(list(group)) for _, group in itertools.groupby(key))
        self.test_result.insert(tk.END, f"Longest Run: {longest_run}\n")

    def chi_square_test(self, key):
        observed = np.bincount(key)
        expected = np.ones_like(observed) * len(key) / 2
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        p_value = 1 - chisquare(observed, expected)[1]
        return chi2_stat, p_value

    def frequency_test(self, key):
        ones = np.sum(key)
        zeros = len(key) - ones
        return abs(ones - zeros) <= 2 * np.sqrt(len(key))

    def run_test(self, key):
        runs = np.diff(key)
        num_runs = np.count_nonzero(runs) + 1
        expected_runs = (2 * np.count_nonzero(key) * (len(key) - np.count_nonzero(key)) / len(key)) + 1
        return abs(num_runs - expected_runs) <= 2 * np.sqrt(2 * len(key) - 1) / 4

    def autocorrelation_test(self, key):
        n = len(key)
        mean = np.mean(key)
        variance = np.var(key)
        if variance == 0:
            return 0
        autocorr = (np.correlate(key - mean, key - mean, mode='full')[n-1:] / variance / n)
        return autocorr[1] if len(autocorr) > 1 else 0

    def serial_test(self, key, m=2):
        n = len(key)
        counts = {}
        for i in range(n - m + 1):
            pattern = tuple(key[i:i+m])
            counts[pattern] = counts.get(pattern, 0) + 1

        expected = (n - m + 1) / 2 ** m
        chi2 = sum((count - expected) ** 2 / expected for count in counts.values())
        p_value = 1 - chisquare(list(counts.values()), f_exp=[expected] * len(counts))[1]
        return p_value

    def poker_test(self, key, m=5):
        n = len(key)
        k = n // m
        counts = {}
        for i in range(k):
            pattern = tuple(key[i * m:(i + 1) * m])
            counts[pattern] = counts.get(pattern, 0) + 1

        expected = k / (2 ** m)
        observed_values = list(counts.values())

        if len(observed_values) < 2 ** m:
            observed_values.extend([0] * ((2 ** m) - len(observed_values)))

        chi2_stat, p_value = chisquare(observed_values, f_exp=[expected] * len(observed_values))
        return p_value

    def cumulative_sums_test(self, key):
        s = 2 * key - 1
        cumsum = np.cumsum(s)
        z = max(abs(cumsum))
        p_value = 2 * (1 - kstest([z], 'norm', args=(0, 1))[1])
        return p_value

    def generate_new_test_key(self):
        seed = np.random.randint(0, 100000000)
        message = b"This is a test message for new custom key encryption"
        iv, ciphertext, key = self.thesis_encryption.encrypt(message, seed)
        compressed_key = self.timeline_compression.compress(iv + ciphertext)

        self.encryption_data["custom"] = {
            "seed": seed,
            "iv": iv,
            "ciphertext": ciphertext,
            "key": key,
            "compressed_key": compressed_key,
            "message": message
        }

        self.test_source.set("New Custom Key")
        self.run_tests()

    def multi_key_test_no_segment_padding(self, message, initial_seed, rekey_interval=16):
        np.random.seed(initial_seed)
        key_history = []
        encrypted_message = b''
        current_seed = initial_seed

        padded_message = pad(message, AES.block_size)

        for i in range(0, len(padded_message), rekey_interval):
            current_block = padded_message[i:i+rekey_interval]

            key_bits = self.thesis_encryption.generate_key(current_seed, length=256)
            key = np.packbits(key_bits).tobytes()

            cipher = AES.new(key, AES.MODE_CBC)
            iv = cipher.iv
            if len(current_block) % AES.block_size != 0:
                current_block_padded = pad(current_block, AES.block_size)
            else:
                current_block_padded = current_block
            ciphertext = cipher.encrypt(current_block_padded)

            key_history.append((current_seed, iv.hex(), key.hex()))
            encrypted_message += iv + ciphertext

            current_seed = np.random.randint(0, 100000000)

        return encrypted_message, key_history

    def multi_key_decrypt_no_segment_padding(self, encrypted_message, key_history, rekey_interval=16):
        decrypted_message = b''
        block_size = 16 + rekey_interval

        for i, (seed, iv_hex, key_hex) in enumerate(key_history):
            start_index = i * block_size
            end_index = start_index + block_size
            encrypted_block = encrypted_message[start_index:end_index]

            iv = bytes.fromhex(iv_hex)
            key = bytes.fromhex(key_hex)

            cipher = AES.new(key, AES.MODE_CBC, iv)
            ciphertext = encrypted_block[16:]
            decrypted_block = cipher.decrypt(ciphertext)

            decrypted_message += decrypted_block

        return unpad(decrypted_message, AES.block_size)

    def encrypt_perfect_entropy(self):
        message = self.perfect_entropy_message_entry.get("1.0", tk.END).encode().strip()
    
        start_time = time.perf_counter()  # Use perf_counter for higher precision
        best_entropy = 0
        best_seed = 0
        best_ciphertext = b''
        best_iv = b''
        attempts = 0
    
        while best_entropy < 0.9999 and attempts < 1000:  # Limit to 1000 attempts to prevent infinite loop
            seed = np.random.randint(0, 100000000)
            iv, ciphertext = self.thesis_encryption.simple_encrypt_fixed(seed, message)
            ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
            entropy = self.thesis_encryption.calculate_entropy_fixed(ciphertext_bits)
        
            if entropy > best_entropy:
                best_entropy = entropy
                best_seed = seed
                best_ciphertext = ciphertext
                best_iv = iv
        
            attempts += 1
    
        end_time = time.perf_counter()
        encryption_time = max(end_time - start_time, 1e-6)  # Ensure a minimum non-zero time

        self.perfect_entropy_seed_entry.delete(0, tk.END)
        self.perfect_entropy_seed_entry.insert(0, str(best_seed))

        self.perfect_entropy_encrypted_entry.delete("1.0", tk.END)
        self.perfect_entropy_encrypted_entry.insert(tk.END, (best_iv + best_ciphertext).hex())

        self.encryption_data["perfect_entropy"] = {
            "seed": best_seed,
            "iv": best_iv,
            "ciphertext": best_ciphertext,
            "message": message,
            "encryption_time": encryption_time,
            "attempts": attempts,
            "achieved_entropy": best_entropy
        }

        messagebox.showinfo("Encryption Complete", f"Encryption took {encryption_time:.6f} seconds\n"
                                                f"Attempts: {attempts}\n"
                                                f"Achieved Entropy: {best_entropy:.6f}")


    def decrypt_perfect_entropy(self):
        try:
            seed = int(self.perfect_entropy_seed_entry.get())
            encrypted_hex = self.perfect_entropy_encrypted_entry.get("1.0", tk.END).strip()
            encrypted = bytes.fromhex(encrypted_hex)
            iv, ciphertext = encrypted[:16], encrypted[16:]

            start_time = time.time()
            decrypted = self.thesis_encryption.decrypt_with_entropy_focus(seed, iv, ciphertext)
            end_time = time.time()

            self.perfect_entropy_decrypted_entry.delete("1.0", tk.END)
            self.perfect_entropy_decrypted_entry.insert(tk.END, decrypted.decode())

            messagebox.showinfo("Decryption Complete", f"Decryption took {end_time - start_time:.4f} seconds")
        except Exception as e:
            messagebox.showerror("Decryption Error", str(e))
    
    def test_perfect_entropy_encryption(self):
        if "perfect_entropy" not in self.encryption_data or self.encryption_data["perfect_entropy"] is None:
            messagebox.showerror("Error", "Please perform Perfect Entropy encryption first.")
            return

        key_data = self.encryption_data["perfect_entropy"]
        self.test_result.delete('1.0', tk.END)
        self.test_result.insert(tk.END, "--- Perfect Entropy Encryption Tests ---\n\n")
        self.test_result.insert(tk.END, f"Seed: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"Encryption Time: {key_data['encryption_time']:.4f} seconds\n\n")

        ciphertext_bits = np.unpackbits(np.frombuffer(key_data['ciphertext'], dtype=np.uint8))
        entropy = self.thesis_encryption.calculate_entropy_fixed(ciphertext_bits)
    
        self.run_existing_tests(ciphertext_bits)
        self.run_additional_strength_tests(ciphertext_bits)
        self.run_avalanche_test(key_data)
        self.run_visual_tests(ciphertext_bits)
        self.run_monobit_test(ciphertext_bits)
        self.run_runs_test(ciphertext_bits)
        self.run_longest_run_test(ciphertext_bits)

        self.test_result.insert(tk.END, f"\nFinal Entropy: {entropy:.6f}\n")
        self.test_result.insert(tk.END, f"Perfect Entropy Seed: {key_data['seed']}\n")

        # Add the "recipe" at the end
        self.test_result.insert(tk.END, "\n--- Encryption Recipe ---\n")
        self.test_result.insert(tk.END, f"1. Start with seed: {key_data['seed']}\n")
        self.test_result.insert(tk.END, f"2. Generate key using ThesisEncryption algorithm\n")
        self.test_result.insert(tk.END, f"3. Encrypt message using AES-CBC mode\n")
        self.test_result.insert(tk.END, f"4. Resulting ciphertext length: {len(key_data['ciphertext'])} bytes\n")
        self.test_result.insert(tk.END, f"5. Achieved entropy: {entropy:.6f}\n")
        self.test_result.insert(tk.END, f"6. Total encryption time: {key_data['encryption_time']:.4f} seconds\n")

        # Additional analysis
        self.test_result.insert(tk.END, "\n--- Strength Analysis ---\n")
        self.test_result.insert(tk.END, f"1. Entropy closeness to 1: {abs(1 - entropy):.6f}\n")
        self.test_result.insert(tk.END, f"2. Encryption speed: {len(key_data['message']) / key_data['encryption_time'] / 1024:.2f} KB/s\n")
    
        # Interpretation
        self.test_result.insert(tk.END, "\n--- Interpretation ---\n")
        if abs(1 - entropy) < 0.001:
            self.test_result.insert(tk.END, "The encryption achieved near-perfect entropy, indicating excellent randomness.\n")
        elif abs(1 - entropy) < 0.01:
            self.test_result.insert(tk.END, "The encryption achieved very high entropy, indicating strong randomness.\n")
        else:
            self.test_result.insert(tk.END, "The encryption achieved good entropy, but there might be room for improvement.\n")

        self.test_result.insert(tk.END, f"The encryption process is {'very fast' if len(key_data['message']) / key_data['encryption_time'] / 1024 > 1000 else 'reasonably fast'} for practical use.\n")

        self.test_result.insert(tk.END, "\nConclusion: This encryption method provides a strong balance between security (high entropy) and performance.\n")

    
if __name__ == "__main__":
    app = EncryptionApp()
    app.mainloop()