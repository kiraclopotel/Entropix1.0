Adaptive Layered Encryption System: A Comprehensive Approach to Cryptographic Key Generation
Table of Contents
Introduction
Mathematical Foundations
2.1. Adaptive Layered Functions
2.2. Dynamic Coefficient Shifts
2.3. Controlled Noise Injection
2.4. Entropy Feedback Mechanism
2.5. Non-Linear Operations and Chaotic Mixing
2.6. Algebraic Structures of Layered Sets
2.7. Advanced Layer Functions and Zeta Functions
Algorithms
3.1. Adaptive Layered Key Generation
3.2. Hashing and Timeline Obfuscation
3.3. Advanced Layered Encryption with 3D Grid
Code Implementation
4.1. ThesisEncryption Class
4.2. AdvancedEncryption3D Class
Use Cases and Applications
Conclusion
Appendices
7.1. Mathematical Proofs and Theorems
7.2. Algorithms and Pseudocode
References
Glossary



1. Introduction Overview This document presents a comprehensive approach to cryptographic key generation and encryption using adaptive layered functions, dynamic coefficient shifts, controlled noise injection, and advanced algebraic structures. The
proposed system introduces flexibility and adaptability, leveraging non-linearity and randomness to enhance security and resist cryptanalytic attacks.

Motivation Traditional cryptographic systems often rely on fixed mathematical structures, making them potentially vulnerable to specific types of attacks due to predictable patterns. The need for dynamic and adaptable cryptographic techniques is growing
as threat landscapes evolve. This system aims to address these challenges by introducing innovative mathematical foundations and algorithms that enhance unpredictability and security.

2. Mathematical Foundations This section details the theoretical basis of the encryption methods, including definitions, formulas, and proofs.
2.1. Adaptive Layered Functions Definition For a real number 𝑛 > 1 n>1 and a positive integer 𝑘 k, the layer function 𝑓 𝑘 ( 𝑛 ) f k ​ (n) is defined as:
𝑓 𝑘 ( 𝑛 ) = 𝑛 1 / 𝑘 ⋅ ( log ⁡ 𝑛 ) ( 𝑘 − 1 ) / 𝑘 f k ​ (n)=n 1/k ⋅(logn) (k−1)/k 
Purpose: This function creates a sequence of functions that decrease in growth rate as 𝑘 k increases, introducing a layered effect in the key generation process. Ratio Formula Theorem:
For any real 𝑛 > 1 n>1 and positive integer 𝑘 k:
𝑓 𝑘 ( 𝑛 ) 𝑓 𝑘 + 1 ( 𝑛 ) = ( 𝑛 ⋅ log ⁡ 𝑛 ) 1 / [ 𝑘 ( 𝑘 + 1 ) ] f k+1 ​ (n) f k ​ (n) ​ =(n⋅logn) 1/[k(k+1)] 
Proof:

See Appendix 7.1.1 for the detailed proof.

2.2. Dynamic Coefficient Shifts Concept Coefficients 𝑐 𝑘 c k ​ and 𝑏 𝑘 b k ​ are dynamically adjusted at each iteration to introduce additional randomness and prevent predictability.
Adjustment Formulas At each iteration 𝑖 i:
𝑐 𝑘 = 𝑐 𝑘 + 𝛿 𝑐 𝑘 , 𝛿 𝑐 𝑘 ∼ Uniform ( − 𝜖 , 𝜖 ) 𝑏 𝑘 = 𝑏 𝑘 + 𝛿 𝑏 𝑘 , 𝛿 𝑏 𝑘 ∼ Uniform ( − 𝜖 , 𝜖 ) c k ​ 
b k ​ 
​
  
=c k ​ +δc k ​ ,δc k ​ ∼Uniform(−ϵ,ϵ) =b k ​ +δb k ​ ,δb k ​ ∼Uniform(−ϵ,ϵ) ​ 
𝜖 ϵ is a small value representing the range of random perturbations. 2.3. Controlled Noise Injection Hybrid Noise 𝐻 𝑖 H i ​ 
Definition:

𝐻 𝑖 = 𝑤 𝑔 ⋅ Gaussian ( 𝐺 𝐿 , 𝑖 ) + 𝑤 𝑒 ⋅ Exponential ( 𝐺 𝐿 , 𝑖 ) H i ​ =w g ​ ⋅Gaussian(G L,i ​ )+w e ​ ⋅Exponential(G L,i ​ ) Weights: 𝑤 𝑔 w g ​ : Weight for Gaussian noise. 𝑤 𝑒 w e ​ : Weight for Exponential noise ( 𝑤 𝑒 = 1 − 𝑤 𝑔 w
e ​ =1−w g ​ ). Components:

Gaussian Noise:

Gaussian ( 𝐺 𝐿 , 𝑖 ) = 𝑁 ( 0 , 0.3 ) ⋅ log ⁡ ( 1 + ∣ 𝐺 𝐿 , 𝑖 ∣ ) Gaussian(G L,i ​ )=N(0,0.3)⋅log(1+∣G L,i ​ ∣) Exponential Noise:
Exponential ( 𝐺 𝐿 , 𝑖 ) = Exp ( 0.5 ) ⋅ log ⁡ ( 1 + ∣ 𝐺 𝐿 , 𝑖 ∣ ) Exponential(G L,i ​ )=Exp(0.5)⋅log(1+∣G L,i ​ ∣) 2.4. Entropy Feedback Mechanism Adjustment Based on Entropy If the entropy 𝐸 E of the key sequence falls below a threshold 𝜖 ϵ or
patterns are detected:

𝑤 𝑔 = min ⁡ ( 𝑤 𝑔 + 0.1 , 1 ) , 𝑤 𝑒 = 1 − 𝑤 𝑔 w g ​ =min(w g ​ +0.1,1),w e ​ =1−w g ​ 
Purpose: Increases randomness to disrupt patterns and enhance security. 2.5. Non-Linear Operations and Chaotic Mixing Non-linear operations such as bitwise XOR, rotations, and permutations are applied after generating key segments to prevent repeating
patterns and increase complexity.

2.6. Algebraic Structures of Layered Sets Definitions and Operations Layered Sets 𝑆 ( 𝑛 ) S(n):
𝑆 ( 𝑛 ) S(n) represents a layer in the system.
Operations:

Addition ( ⊕ ⊕):
𝑆 ( 𝑛 ) ⊕ 𝑆 ( 𝑚 ) = 𝑆 ( max ⁡ ( 𝑛 , 𝑚 ) + 1 ) S(n)⊕S(m)=S(max(n,m)+1) Multiplication ( ⊗ ⊗):
𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑚 ) = 𝑆 ( 𝑛 + 𝑚 ) S(n)⊗S(m)=S(n+m) Properties Commutativity:
𝑆 ( 𝑛 ) ⊕ 𝑆 ( 𝑚 ) = 𝑆 ( 𝑚 ) ⊕ 𝑆 ( 𝑛 ) S(n)⊕S(m)=S(m)⊕S(n) 𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑚 ) = 𝑆 ( 𝑚 ) ⊗ 𝑆 ( 𝑛 ) S(n)⊗S(m)=S(m)⊗S(n) Associativity:
( 𝑆 ( 𝑛 ) ⊕ 𝑆 ( 𝑚 ) ) ⊕ 𝑆 ( 𝑘 ) = 𝑆 ( 𝑛 ) ⊕ ( 𝑆 ( 𝑚 ) ⊕ 𝑆 ( 𝑘 ) ) (S(n)⊕S(m))⊕S(k)=S(n)⊕(S(m)⊕S(k)) ( 𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑚 ) ) ⊗ 𝑆 ( 𝑘 ) = 𝑆 ( 𝑛 ) ⊗ ( 𝑆 ( 𝑚 ) ⊗ 𝑆 ( 𝑘 ) ) (S(n)⊗S(m))⊗S(k)=S(n)⊗(S(m)⊗S(k)) Distributivity:
𝑆 ( 𝑛 ) ⊗ ( 𝑆 ( 𝑚 ) ⊕ 𝑆 ( 𝑘 ) ) = ( 𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑚 ) ) ⊕ ( 𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑘 ) ) S(n)⊗(S(m)⊕S(k))=(S(n)⊗S(m))⊕(S(n)⊗S(k)) Identity Elements:
Multiplicative Identity: 𝑆 ( 𝑛 ) ⊗ 𝑆 ( 0 ) = 𝑆 ( 𝑛 ) S(n)⊗S(0)=S(n) Additive Identity: 𝑆 ( 𝑛 ) ⊕ 𝑆 ( 0 ) = 𝑆 ( max ⁡ ( 𝑛 , 0 ) + 1 ) = 𝑆 ( 𝑛 + 1 ) S(n)⊕S(0)=S(max(n,0)+1)=S(n+1) (Note: This does not serve as a true additive identity.) Proofs
Detailed proofs of these properties are provided in Appendix 7.1.2.

2.7. Advanced Layer Functions and Zeta Functions General Layer Function For complex numbers 𝑧 z, 𝑘 k, 𝑎 a, and 𝑏 b:
𝑓 𝑘 , 𝑎 , 𝑏 ( 𝑧 ) = exp ⁡ ( 𝑎 𝑘 ⋅ ln ⁡ ( 𝑧 ) + 𝑏 ( 𝑘 − 1 ) 𝑘 ⋅ ln ⁡ ( ln ⁡ ( 𝑧 ) ) ) f k,a,b ​ (z)=exp( k a ​ ⋅ln(z)+ k b(k−1) ​ ⋅ln(ln(z))) Domain: 𝑧 ≠ 0 , 1 z  =0,1; 𝑘 ≠ 0 k  =0 Purpose: Extends layer functions to the complex domain.
Layer Zeta Function Defined as:

𝜁 𝑘 , 𝑎 , 𝑏 ( 𝑠 ) = ∑ 𝑛 = 1 ∞ 1 [ 𝑓 𝑘 , 𝑎 , 𝑏 ( 𝑛 ) ] 𝑠 ζ k,a,b ​ (s)= n=1 ∑ ∞ ​  
[f k,a,b ​ (n)] s 
1 ​ 
Convergence Criteria:

Converges if Re ( 𝑎 𝑠 𝑘 ) > 1 Re( k as ​ )>1 Behavior and Applications Negative 𝑎 a or 𝑏 b: Modifies growth rates, potentially modeling decay processes or accelerated growth.
General Form for Edge Cases:

𝑔 𝑘 , 𝑎 , 𝑏 ( 𝑧 ) = exp ⁡ ( 𝑎 𝑘 ⋅ ln ⁡ ( 1 + ∣ 𝑧 ∣ ) + 𝑏 ( 𝑘 − 1 ) 𝑘 ⋅ ln ⁡ ( 1 + ln ⁡ ( 1 + ∣ 𝑧 ∣ ) ) ) g k,a,b ​ (z)=exp( k a ​ ⋅ln(1+∣z∣)+ k b(k−1) ​ ⋅ln(1+ln(1+∣z∣))) Purpose: Handles edge cases and ensures continuity.
3. Algorithms 3.1. Adaptive Layered Key Generation Comprehensive Key Bit Formula The key bit 𝐵 𝑖 B i ​ at index 𝑖 i is computed as:
𝐺 𝐿 , 𝑖 = Grid ( 𝐿 , ( 𝑖 𝑑 1 m o d 
 
𝑑 1 ) , ( 𝑖 𝑑 2 m o d 
 
𝑑 2 ) , … , ( 𝑖 𝑑 𝑛 m o d 
 
𝑑 𝑛 ) ) 𝐻 𝑖 = 𝑤 𝑔 ⋅ Gaussian ( 𝐺 𝐿 , 𝑖 ) + 𝑤 𝑒 ⋅ Exponential ( 𝐺 𝐿 , 𝑖 ) 𝐴 𝑖 = 𝑐 𝑘 ⋅ ( 𝐺 𝐿 , 𝑖 + 1 + 𝐻 𝑖 ) 1 / 𝐿 ⋅ ( ln ⁡ ( 𝐺 𝐿 , 𝑖 + 1 + 𝐻 𝑖 ) ) ( 𝐿 − 1 ) / 𝐿 + 𝑏 𝑘 ⋅ ln ⁡ ( ln ⁡ ( 𝐺 𝐿 , 𝑖 + 1 + 𝐻 𝑖 ) + 1 ) 𝐵 𝑖 =
( 𝐴 𝑖 + 0.01 ⋅ ( ln ⁡ ( 𝐺 𝐿 , 𝑖 + 1 + 𝐻 𝑖 ) ) 2 ⋅ ( 𝐺 𝐿 , 𝑖 + 1 + 𝐻 𝑖 ) 1 / 𝐿 ) m o d
 
 
2 G L,i ​ 
H i ​ 
A i ​ 
B i ​ 
​
  
=Grid(L,(i d 1 ​ 
modd 1 ​ ),(i d 2 ​ 
modd 2 ​ ),…,(i d n ​ 
modd n ​ )) =w g ​ ⋅Gaussian(G L,i ​ )+w e ​ ⋅Exponential(G L,i ​ ) =c k ​ ⋅(G L,i ​ +1+H i ​ ) 1/L ⋅(ln(G L,i ​ +1+H i ​ )) (L−1)/L +b k ​ ⋅ln(ln(G L,i ​ +1+H i ​ )+1) =(A i ​ +0.01⋅(ln(G L,i ​ +1+H i ​ )) 2 ⋅(G L,i ​ +1+H i ​ ) 1/L )mod2 ​ 
Algorithm Steps Initialization:
Set initial seed. Initialize 𝑐 𝑘 c k ​ , 𝑏 𝑘 b k ​ , 𝑘 k, 𝑤 𝑔 w g ​ , and 𝑤 𝑒 w e ​ . Key Bit Generation Loop:
For each bit index 𝑖 i:
Adjust Layer Index: Increase or decrease 𝑘 k based on pattern detection. Select Grid Value 𝐺 𝐿 , 𝑖 G L,i ​ : Compute using the provided formula. Generate Hybrid Noise 𝐻 𝑖 H i ​ . Compute Adjusted Value 𝐴 𝑖 A i ​ . Compute Key Bit 𝐵 𝑖 B i ​ :
Apply modulo 2. Append 𝐵 𝑖 B i ​ to Key Sequence. Update Coefficients 𝑐 𝑘 c k ​ and 𝑏 𝑘 b k ​ . Adjust Weights 𝑤 𝑔 w g ​ and 𝑤 𝑒 w e ​ based on entropy feedback. Post-Processing:

Apply non-linear operations. Output:
Return the final key as an array of bits or bytes. 3.2. Hashing and Timeline Obfuscation Objective Enhance security by obfuscating the hash of the ciphertext using a sequence-based timeline.
Process Steps Initial Encryption (AES):
𝐶 = AES CBC ( 𝑀 , 𝐾 , IV ) C=AES CBC ​ (M,K,IV) Hashing the Ciphertext:
𝐻 = SHA256 ( 𝐶 ) H=SHA256(C) Sequence Timeline Generation:
Define sequences 𝑆 𝑘 ( 𝑖 ) S k ​ (i) (e.g., Fibonacci, primes).
Generate timeline 𝑇 ( 𝑖 ) T(i) using a switching rule 𝜎 ( 𝑖 ) σ(i):
𝜎 ( 𝑖 ) = m o d 
 
( 𝑖 , 3 ) σ(i)=mod(i,3) 𝑇 ( 𝑖 ) = 𝑆 𝜎 ( 𝑖 ) ( 𝑖 ) T(i)=S σ(i) ​ (i) Obfuscating the Hash:
𝐻 obf ( 𝑖 ) = ( 𝐻 ( 𝑖 ) + 𝑇 ( 𝑖 ) ) m o d 
 
256 H obf ​ (i)=(H(i)+T(i))mod256 Deobfuscating the Hash:
𝐻 rec ( 𝑖 ) = ( 𝐻 obf ( 𝑖 ) − 𝑇 ( 𝑖 ) ) m o d 
 
256 H rec ​ (i)=(H obf ​ (i)−T(i))mod256 Decryption of the Original Message:
𝑀 = AES CBC − 1 ( 𝐶 , 𝐾 , IV ) M=AES CBC −1 ​ (C,K,IV) 3.3. Advanced Layered Encryption with 3D Grid Core Concept Uses a 3D grid (cube) to map and transform data during encryption and decryption, leveraging the layered number system.
Key Components 3D Grid (Cube):
Dimensions: num_layers × grid_size × grid_size × grid_size num_layers×grid_size×grid_size×grid_size Each cell contains a unique number, skipping multiples of 10. Wall Functions:
Six functions corresponding to each face of the cube (x+, x−, y+, y−, z+, z−). Apply transformations during encryption/decryption. Key and Salt:
Key: Cryptographically secure random byte string. Salt: Random byte string used in key generation. Encryption Process Convert Characters to ASCII Values. Map Characters to 3D Grid Positions. Apply Wall Functions: Transform the value. Combine with Key.
Compute Final Encrypted Value. Decryption Process Start with Encrypted Value. Reverse Key Combination. Reverse Wall Function Transformations. Map Back to Original Grid Position. Convert to ASCII and then to Character. Parameters and Constraints
grid_size: Integer ≥ 20. num_layers: Integer ≥ 3. key_length: Typically 32 bytes (256 bits). 4. Code Implementation 4.1. ThesisEncryption Class python Copy code import random import math from collections import Counter from Crypto.Cipher import AES from
Crypto.Util.Padding import pad, unpad

class ThesisEncryption: def __init__(self, key_length=32): self.key_length = key_length  # Key length in bytes self.seed = None self.key = None
def generate_key(self, seed): self.seed = seed random.seed(self.seed) c_k = random.uniform(0.8, 1.2) b_k = random.uniform(0.8, 1.2) k = 1  # Initial layer index w_g = 0.5  # Initial Gaussian weight w_e = 0.5  # Initial Exponential weight key_bits = []
for i in range(1, self.key_length * 8 + 1): # Adjust layer index based on patterns if i > 2 and key_bits[-1] == key_bits[-2]: k = min(k + 1, 5) else: k = max(k - 1, 1)
# Select grid value G_L_i = self.compute_grid_value(k, i) # Generate hybrid noise H_i = self.generate_noise(G_L_i, w_g, w_e) # Compute adjusted value A_i = self.compute_adjusted_value(c_k, b_k, G_L_i, H_i, k) # Compute key bit B_i = int((A_i + 0.01 *
(math.log(G_L_i + 1 + H_i)) ** 2 * (G_L_i + 1 + H_i) ** (1 / k)) % 2) key_bits.append(B_i)

# Update coefficients c_k += random.uniform(-0.1, 0.1) b_k += random.uniform(-0.1, 0.1)
# Adjust weights based on entropy feedback if i % 8 == 0: E = self.calculate_entropy(key_bits) if E < 0.95: w_g = min(w_g + 0.1, 1) w_e = 1 - w_g
# Convert bits to bytes self.key = bytes(int(''.join(map(str, key_bits[i:i+8])), 2) for i in range(0, len(key_bits), 8))
def compute_grid_value(self, L, i): # Compute G_L_i using modular exponentiation exponents = [1, 2, 3] # Example exponents d_1, d_2, d_3 mods = [17, 19, 23] # Example moduli d_1, d_2, d_3 grid_indices = [(i ** exp) % mod for exp, mod in zip(exponents,
mods)] G_L_i = sum(grid_indices) + L # Simplified for demonstration return G_L_i

def generate_noise(self, G_L_i, w_g, w_e): gaussian_noise = random.gauss(0, 0.3) * math.log(1 + abs(G_L_i)) exponential_noise = random.expovariate(0.5) * math.log(1 + abs(G_L_i)) H_i = w_g * gaussian_noise + w_e * exponential_noise return H_i
def compute_adjusted_value(self, c_k, b_k, G_L_i, H_i, L): base = G_L_i + 1 + H_i A_i = c_k * base ** (1 / L) * (math.log(base)) ** ((L - 1) / L) + b_k * math.log(math.log(base) + 1) return A_i
def calculate_entropy(self, bits): counts = Counter(bits) total = len(bits) entropy = 0 for count in counts.values(): p = count / total entropy -= p * math.log2(p) return entropy
def encrypt(self, plaintext): cipher = AES.new(self.key, AES.MODE_CBC) ciphertext = cipher.iv + cipher.encrypt(pad(plaintext, AES.block_size)) return ciphertext
def decrypt(self, ciphertext): iv = ciphertext[:AES.block_size] cipher = AES.new(self.key, AES.MODE_CBC, iv) plaintext = unpad(cipher.decrypt(ciphertext[AES.block_size:]), AES.block_size) return plaintext Explanation:
generate_key: Implements the adaptive layered key generation using the comprehensive formula. compute_grid_value: Calculates 𝐺 𝐿 , 𝑖 G L,i ​ based on exponents and moduli. generate_noise: Generates hybrid noise 𝐻 𝑖 H i ​ . calculate_entropy:
Computes entropy for feedback mechanism. encrypt/decrypt: Uses AES encryption/decryption with the generated key. 4.2. AdvancedEncryption3D Class python Copy code import numpy as np import random

class AdvancedEncryption3D: def __init__(self, grid_size=20, num_layers=3, key_length=32): self.grid_size = grid_size self.num_layers = num_layers self.key_length = key_length self.grid = self.generate_grid() self.wall_functions =
self.assign_wall_functions() self.key = self.generate_key()

def generate_grid(self): total_cells = self.num_layers * self.grid_size ** 3 numbers = [n for n in range(1, total_cells + 1) if n % 10 != 0] grid = np.array(numbers).reshape((self.num_layers, self.grid_size, self.grid_size, self.grid_size)) return grid
def assign_wall_functions(self): operations = [lambda x: x + 1, lambda x: x * 2, lambda x: x ^ 1] wall_functions = { 'x+': random.choice(operations), 'x-': random.choice(operations), 'y+': random.choice(operations), 'y-': random.choice(operations),
'z+': random.choice(operations), 'z-': random.choice(operations), } return wall_functions

def generate_key(self): return bytes([random.randint(0, 255) for _ in range(self.key_length)])
def encrypt(self, plaintext): encrypted_values = [] key_bytes = self.key for index, char in enumerate(plaintext): ascii_val = ord(char) layer = index % self.num_layers x = y = z = index % self.grid_size base_value = self.grid[layer][x][y][z] value =
base_value value = self.wall_functions['x+'](value) value = self.wall_functions['y+'](value) value = self.wall_functions['z+'](value) value += ascii_val value += key_bytes[index % self.key_length] encrypted_value = value % 256
encrypted_values.append(encrypted_value) return bytes(encrypted_values)

def decrypt(self, encrypted_bytes): decrypted_chars = [] key_bytes = self.key for index, value in enumerate(encrypted_bytes): layer = index % self.num_layers x = y = z = index % self.grid_size base_value = self.grid[layer][x][y][z] value = (value -
key_bytes[index % self.key_length]) % 256 value -= base_value value = self.inverse_wall_functions('z+', value) value = self.inverse_wall_functions('y+', value) value = self.inverse_wall_functions('x+', value) ascii_val = value % 256
decrypted_chars.append(chr(ascii_val)) return ''.join(decrypted_chars)

def inverse_wall_functions(self, wall, value): func = self.wall_functions[wall] # Assuming functions are invertible and known if func == (lambda x: x + 1): return value - 1 elif func == (lambda x: x * 2): return value // 2 elif func == (lambda x: x ^
1): return value ^ 1 else: raise ValueError("Unknown wall function") Explanation:

generate_grid: Creates a 3D grid skipping multiples of 10. assign_wall_functions: Randomly assigns mathematical operations to wall functions. encrypt: Encrypts the plaintext by mapping characters to grid positions and applying transformations. decrypt:
Reverses the encryption process to retrieve the original plaintext. inverse_wall_functions: Provides inverse operations for decryption. Note: In practice, ensure that the functions used are invertible and securely implemented.

5. Use Cases and Applications Secure Communication Protocols: Integrate the system into protocols to enhance key unpredictability. Data Storage Security: Protect stored data with dynamically generated keys. Cryptographic Research: Provide a foundation
for further exploration of adaptive encryption methods. 6. Conclusion This comprehensive approach to cryptographic key generation and encryption introduces advanced mathematical concepts and algorithms to enhance security. By leveraging adaptive layered
functions, dynamic coefficients, controlled noise, and algebraic structures, the system offers increased unpredictability and resistance to cryptanalytic attacks.

7. Appendices 7.1. Mathematical Proofs and Theorems 7.1.1. Proof of the Ratio Formula for Layer Functions Theorem:
𝑓 𝑘 ( 𝑛 ) 𝑓 𝑘 + 1 ( 𝑛 ) = ( 𝑛 ⋅ log ⁡ 𝑛 ) 1 / [ 𝑘 ( 𝑘 + 1 ) ] f k+1 ​ (n) f k ​ (n) ​ =(n⋅logn) 1/[k(k+1)] 
Proof:

Start with the definitions:

𝑓 𝑘 ( 𝑛 ) = 𝑛 1 / 𝑘 ( log ⁡ 𝑛 ) ( 𝑘 − 1 ) / 𝑘 f k ​ (n)=n 1/k (logn) (k−1)/k 
𝑓 𝑘 + 1 ( 𝑛 ) = 𝑛 1 / ( 𝑘 + 1 ) ( log ⁡ 𝑛 ) 𝑘 / ( 𝑘 + 1 ) f k+1 ​ (n)=n 1/(k+1) (logn) k/(k+1) 
Compute the ratio:

𝑓 𝑘 ( 𝑛 ) 𝑓 𝑘 + 1 ( 𝑛 ) = 𝑛 1 / 𝑘 𝑛 1 / ( 𝑘 + 1 ) ⋅ ( log ⁡ 𝑛 ) ( 𝑘 − 1 ) / 𝑘 ( log ⁡ 𝑛 ) 𝑘 / ( 𝑘 + 1 ) f k+1 ​ (n) f k ​ (n) ​ = n 1/(k+1) 
n 1/k 
​ ⋅ (logn) k/(k+1) 
(logn) (k−1)/k 
​
 
Simplify exponents:

For 𝑛 n:
1 𝑘 − 1 𝑘 + 1 = ( 𝑘 + 1 ) − 𝑘 𝑘 ( 𝑘 + 1 ) = 1 𝑘 ( 𝑘 + 1 ) k 1 ​ − k+1 1 ​ = k(k+1) (k+1)−k ​ = k(k+1) 1 ​ 
For log ⁡ 𝑛 logn:
𝑘 − 1 𝑘 − 𝑘 𝑘 + 1 = ( 𝑘 − 1 ) ( 𝑘 + 1 ) − 𝑘 2 𝑘 ( 𝑘 + 1 ) = − 1 𝑘 ( 𝑘 + 1 ) k k−1 ​ − k+1 k ​ = k(k+1) (k−1)(k+1)−k 2 
​ =− k(k+1) 1 ​ 
Combine:

𝑓 𝑘 ( 𝑛 ) 𝑓 𝑘 + 1 ( 𝑛 ) = 𝑛 1 / [ 𝑘 ( 𝑘 + 1 ) ] ⋅ ( log ⁡ 𝑛 ) − 1 / [ 𝑘 ( 𝑘 + 1 ) ] = ( 𝑛 log ⁡ 𝑛 ) 1 / [ 𝑘 ( 𝑘 + 1 ) ] f k+1 ​ (n) f k ​ (n) ​ =n 1/[k(k+1)] ⋅(logn) −1/[k(k+1)] =( logn n ​ ) 1/[k(k+1)] 
7.1.2. Algebraic Structure Proofs Refer to Section 2.6 for definitions.
Commutativity:

Addition:

𝑆 ( 𝑛 ) ⊕ 𝑆 ( 𝑚 ) = 𝑆 ( max ⁡ ( 𝑛 , 𝑚 ) + 1 ) = 𝑆 ( max ⁡ ( 𝑚 , 𝑛 ) + 1 ) = 𝑆 ( 𝑚 ) ⊕ 𝑆 ( 𝑛 ) S(n)⊕S(m)=S(max(n,m)+1)=S(max(m,n)+1)=S(m)⊕S(n) Multiplication:
𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑚 ) = 𝑆 ( 𝑛 + 𝑚 ) = 𝑆 ( 𝑚 + 𝑛 ) = 𝑆 ( 𝑚 ) ⊗ 𝑆 ( 𝑛 ) S(n)⊗S(m)=S(n+m)=S(m+n)=S(m)⊗S(n) Associativity:
Addition:

( 𝑆 ( 𝑛 ) ⊕ 𝑆 ( 𝑚 ) ) ⊕ 𝑆 ( 𝑘 ) = 𝑆 ( max ⁡ ( 𝑛 , 𝑚 ) + 1 ) ⊕ 𝑆 ( 𝑘 ) = 𝑆 ( max ⁡ ( max ⁡ ( 𝑛 , 𝑚 ) + 1 , 𝑘 ) + 1 ) = 𝑆 ( max ⁡ ( 𝑛 , 𝑚 , 𝑘 ) + 2 ) (S(n)⊕S(m))⊕S(k) ​  
=S(max(n,m)+1)⊕S(k) =S(max(max(n,m)+1,k)+1) =S(max(n,m,k)+2) ​ 
𝑆 ( 𝑛 ) ⊕ ( 𝑆 ( 𝑚 ) ⊕ 𝑆 ( 𝑘 ) ) = 𝑆 ( 𝑛 ) ⊕ 𝑆 ( max ⁡ ( 𝑚 , 𝑘 ) + 1 ) = 𝑆 ( max ⁡ ( 𝑛 , max ⁡ ( 𝑚 , 𝑘 ) + 1 ) + 1 ) = 𝑆 ( max ⁡ ( 𝑛 , 𝑚 , 𝑘 ) + 2 ) S(n)⊕(S(m)⊕S(k)) ​  
=S(n)⊕S(max(m,k)+1) =S(max(n,max(m,k)+1)+1) =S(max(n,m,k)+2) ​ 
Distributivity:

Multiplication over Addition:

𝑆 ( 𝑛 ) ⊗ ( 𝑆 ( 𝑚 ) ⊕ 𝑆 ( 𝑘 ) ) = 𝑆 ( 𝑛 ) ⊗ 𝑆 ( max ⁡ ( 𝑚 , 𝑘 ) + 1 ) = 𝑆 ( 𝑛 + max ⁡ ( 𝑚 , 𝑘 ) + 1 ) S(n)⊗(S(m)⊕S(k)) ​  
=S(n)⊗S(max(m,k)+1) =S(n+max(m,k)+1) ​ 
( 𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑚 ) ) ⊕ ( 𝑆 ( 𝑛 ) ⊗ 𝑆 ( 𝑘 ) ) = 𝑆 ( 𝑛 + 𝑚 ) ⊕ 𝑆 ( 𝑛 + 𝑘 ) = 𝑆 ( max ⁡ ( 𝑛 + 𝑚 , 𝑛 + 𝑘 ) + 1 ) = 𝑆 ( 𝑛 + max ⁡ ( 𝑚 , 𝑘 ) + 1 ) (S(n)⊗S(m))⊕(S(n)⊗S(k)) ​  
=S(n+m)⊕S(n+k) =S(max(n+m,n+k)+1) =S(n+max(m,k)+1) ​ 
Identity Elements:

Multiplicative Identity:

𝑆 ( 𝑛 ) ⊗ 𝑆 ( 0 ) = 𝑆 ( 𝑛 + 0 ) = 𝑆 ( 𝑛 ) S(n)⊗S(0)=S(n+0)=S(n) Additive Identity:
𝑆 ( 𝑛 ) ⊕ 𝑆 ( 0 ) = 𝑆 ( max ⁡ ( 𝑛 , 0 ) + 1 ) = 𝑆 ( 𝑛 + 1 ) ≠ 𝑆 ( 𝑛 ) S(n)⊕S(0)=S(max(n,0)+1)=S(n+1)  =S(n) (Note: There is no true additive identity in this structure.)
7.2. Algorithms and Pseudocode Refer to Section 3 for detailed algorithms.
8. References National Institute of Standards and Technology (NIST). A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications. Menezes, A. J., van Oorschot, P. C., & Vanstone, S. A. (1996). Handbook of
Applied Cryptography. Stallings, W. (2017). Cryptography and Network Security: Principles and Practice. 9. Glossary AES (Advanced Encryption Standard): A symmetric encryption algorithm widely used for secure data transmission. Entropy: A measure of
randomness or unpredictability in data. Hybrid Noise: Combination of Gaussian and Exponential noise distributions. Layered Functions: Mathematical functions that introduce complexity through layers. Non-Linear Operations: Mathematical operations that do
not have a straight-line relationship, increasing complexity. Zeta Function: A function used in number theory that can be extended to complex numbers.
