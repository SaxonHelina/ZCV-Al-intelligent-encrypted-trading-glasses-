AI Assistant with Voice and Gesture Control

This project enables interaction with an AI assistant through voice commands and hand gestures, providing a hands-free experience for tasks such as report queries and strategy adjustments.

Features:

Voice Recognition: Processes and understands spoken commands.
Gesture Detection: Recognizes specific hand gestures for control.
Real-Time Interaction: Provides immediate responses and feedback.
AI Training Strategy: Incorporates machine learning models to enhance decision-making processes.
Hardware Functionality: Integrates with hardware components for a seamless user experience.
Blockchain Integration: Utilizes blockchain technology for secure and transparent data management.
Related Projects:

CVM Runtime (AI Container): A containerized environment for deploying AI models.
File Storage: A decentralized file storage system.
AI Wrapper: Provides a fixed API for inference and file storage.
PoW (Cortex Cuckoo Cycle): A proof-of-work algorithm for blockchain.
Rosetta: A tool for cross-chain interoperability.
Docker: Containerization platform for deploying applications.
Robot: A framework for building AI-powered robots.
System Requirements:

Operating System: Ubuntu 18.04 or higher.
CPU: x64 architecture with support for AVX, AVX2, and AVX512 instructions.
Memory: At least 16GB of RAM.
Storage: 500GB SSD for development; 1TB SSD with a minimum of 3k dedicated IOPS for production.
GPU (Optional): NVIDIA driver version 470.63.01 or higher; CUDA Toolkit version 12.3 or higher.
Setup:

Clone the Repository:

bash
git clone https://github.com/yourusername/ai-assistant.git
Install Dependencies:

bash
pip install -r requirements.txt
Run the Application:

bash
python ai_assistant.py
Usage:

Voice Commands:

"Report": Fetches the latest financial report.
"Strategy": Adjusts the trading strategy.
"Exit": Closes the AI assistant.
Hand Gestures:

Thumbs Up: Confirms an action.
Swipe Left: Switches modes.
Open Hand: Stops the AI assistant.
AI Training Strategy:

The AI assistant employs machine learning models to enhance its decision-making capabilities. These models are trained on historical data to predict market trends and suggest optimal strategies. The training process involves:

Data Collection: Gathering historical market data.
Preprocessing: Cleaning and normalizing the data.
Model Training: Using algorithms like neural networks to train the model.
Evaluation: Assessing the model's performance and accuracy.
Hardware Functionality:

The AI assistant integrates with hardware components such as cameras and microphones to facilitate voice and gesture recognition. The hardware setup includes:

Camera: Captures real-time video for gesture detection.
Microphone: Records audio for voice command processing.
Processing Unit: Handles data processing and AI computations.
Blockchain Integration:

To ensure secure and transparent data management, the AI assistant utilizes blockchain technology. This integration provides:

Data Integrity: Ensures that data cannot be tampered with.
Transparency: Allows users to view transaction histories.
Security: Protects sensitive information through encryption.
Note:

Ensure your system has a webcam for gesture detection and a microphone for voice commands.

AI wrapper (Fixed API for inference and file storage)
https://github.com/CortexFoundation/inference

PoW (Cortex Cuckoo cycle)
https://github.com/CortexFoundation/solution

Rosseta
https://github.com/CortexFoundation/rosetta-cortex

Docker
https://github.com/CortexFoundation/docker

Robot
https://github.com/CortexFoundation/robot

System Requirements
**** x64 support ****
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single pti ibrs ibpb stibp fsgsbase bmi1 avx2 smep bmi2 erms invpcid xsaveopt
For example

cat /proc/cpuinfo 
Support
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 63
model name	: Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
stepping	: 2
microcode	: 0x1
cpu MHz		: 2494.224
cache size	: 30720 KB
physical id	: 0
siblings	: 2
core id		: 0
cpu cores	: 1
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single pti ibrs ibpb stibp fsgsbase bmi1 avx2 smep bmi2 erms invpcid xsaveopt
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit
bogomips	: 4988.44
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 48 bits virtual
Not Support
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              32
On-line CPU(s) list: 0-31
Thread(s) per core:  2
Core(s) per socket:  16
Socket(s):           1
NUMA node(s):        2
Vendor ID:           AuthenticAMD
CPU family:          23
Model:               1
Model name:          AMD EPYC 7571
Stepping:            2
CPU MHz:             2534.021
BogoMIPS:            4399.86
Hypervisor vendor:   KVM
Virtualization type: full
L1d cache:           32K
L1i cache:           64K
L2 cache:            512K
L3 cache:            8192K
NUMA node0 CPU(s):   0-7,16-23
NUMA node1 CPU(s):   8-15,24-31
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid amd_dcm aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch topoext perfctr_core vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr arat npt nrip_save
ubuntu
Cortex node is developed in Ubuntu 18.04 x64 + CUDA 9.2 + NVIDIA Driver 396.37 environment, with CUDA Compute capability >= 6.1. Latest Ubuntu distributions are also compatible, but not fully tested. Recommend:

cmake 3.11.0+
wget https://cmake.org/files/v3.11/cmake-3.11.0-rc4-Linux-x86_64.tar.gz
tar zxvf cmake-3.11.0-rc4-Linux-x86_64.tar.gz
sudo mv cmake-3.11.0-rc4-Linux-x86_64  /opt/cmake-3.11
sudo ln -sf /opt/cmake-3.11/bin/*  /usr/bin/

sudo apt-get install make
go 1.20.+
wget https://go.dev/dl/go1.20.2.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.20.2.linux-amd64.tar.gz
echo 'export PATH="$PATH:/usr/local/go/bin"' >> ~/.bashrc
source ~/.bashrc
gcc/g++ 5.4+
sudo apt install gcc
sudo apt install g++
cuda 9.2+ (if u have gpu)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
nvidia driver 396.37+ reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#major-components
ubuntu 18.04+
*centos (not recommended)
Recommend:

cmake 3.11.0+
yum install cmake3
go 1.20.+
gcc/g++ 5.4+ reference: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version
cuda 10.1+ (if u have gpu)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
nvidia driver 418.67+
centos 7.6
Cortex Full Node
Compile Source Code (8G+ Memory suggested)
git clone --recursive https://github.com/CortexFoundation/CortexTheseus.git
cd CortexTheseus
make clean && make -j$(nproc)
It is important to pass this check of libcvm_runtime.so
ldd plugins/libcvm_runtime.so

linux-vdso.so.1 =>  (0x00007ffe107fa000)
libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f250e6a8000)
libm.so.6 => /lib64/libm.so.6 (0x00007f250e3a6000)
libgomp.so.1 => /lib64/libgomp.so.1 (0x00007f250e180000)
libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f250df6a000)
libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f250dd4e000)
libc.so.6 => /lib64/libc.so.6 (0x00007f250d980000)
/lib64/ld-linux-x86-64.so.2 (0x00007f250ed35000)
(If failed, run rm -rf cvm-runtime && git submodule init && git submodule update and try again)
