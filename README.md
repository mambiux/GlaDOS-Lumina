# L.U.M.I.N.A: Linguistic Understanding Machine Intelligence Neural Agent

Greetings! This project, **L.U.M.I.N.A.**, is inspired by the incredible groundwork laid by the GLaDOS Personality Core project. Huge thanks to the original creator and community for their brilliant efforts, which paved the way for this evolution. L.U.M.I.N.A. aims to take the concept further, blending advanced artificial intelligence, personality, and multi-agent capabilities into a robust AI assistant capable of self-improvement and, perhaps one day, sentience.

---

## Introduction
L.U.M.I.N.A. is an advanced AI assistant designed to:
- Host a dynamic **personality core** capable of **learning, evolving, and adapting** through interaction.
- Integrate a **multi-agent architecture**, where specialized agent cores ("Experts") collaborate to deliver precise and context-aware responses.
- Utilize **WebSockets** for real-time, bidirectional communication, enabling users to chat with L.U.M.I.N.A. from anywhere.
- Serve as a platform for **cutting-edge AI experiments**, including integration with large language models (LLMs) and modular expansion for future AI technologies.

---

## Key Features
- **Multi-Agentic System**:
  L.U.M.I.N.A. employs an LLM router to delegate queries to specialized agent cores, each tailored to specific tasks or domains.

- **WebSocket Communication**:
  Provides direct, real-time access to L.U.M.I.N.A. via mobile or remote systems, making it a constant companion.

- **Self-Evolution**:
  Designed to improve over time through memory systems, feedback loops, and modular upgrades, with the potential for adaptive behaviors.

- **Linux-Only Installation**:
  Optimized for Linux systems to ensure performance and simplicity in development.

---

## Software Architecture
L.U.M.I.N.A.'s software architecture is built on a foundation of:

- **LLM Router**:
  Routes user prompts to relevant agent cores, each operating as a specialized expert.

- **Memory Systems**:
  Implements short-term, medium-term, and long-term memory to contextualize interactions and support continuous learning.

- **Streaming Voice Interaction**:
  Combines low-latency voice detection with transcription and text-to-speech systems for fluid communication.

- **WebSocket Server**:
  Facilitates live interactions with L.U.M.I.N.A., enabling seamless communication from any device with an internet connection.

### Subgoals
1. Minimal dependencies for a lean system capable of running on constrained hardware.
2. Modular architecture to allow integration of future AI advancements and expansion of capabilities.
3. Ensure robust privacy and security features to protect user data and interactions.

---

## Hardware Recommendations
While L.U.M.I.N.A. is designed for flexibility, the following hardware setup is recommended for optimal performance:

- **CPU**: Modern multi-core processors.
- **RAM**: At least 8GB (16GB or more for larger models).
- **Storage**: SSD with sufficient space for models and logs.
- **Network**: Reliable internet connection for WebSocket-based communication.

---

## Installation Instructions (Linux Only)

### Prerequisites
1. **Python 3.12+**
   Install via your package manager or [Python.org](https://www.python.org/downloads/).

2. **Git**
   ```bash
   sudo apt install git
   ```

3. **Dependencies**
   Install necessary packages:
   ```bash
   sudo apt install build-essential espeak-ng libespeak-ng-dev
   ```

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/lumina.git
   cd lumina
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Models**:
   Place the required models in the `.models` directory:
   - [Voice Recognition Model](https://huggingface.co/distil-whisper/distil-medium.en)
   - [LLM Model](https://huggingface.co/Meta/Meta-Llama-3-8B-Instruct)

4. **Configure WebSockets**:
   Edit the `lumina_config.yaml` file to set up your WebSocket server and client configurations.

5. **Run L.U.M.I.N.A.**:
   ```bash
   python lumina.py
   ```

---

## Roadmap
1. **Enhanced Personality Core**:
   Develop deeper emotional and cognitive abilities.

2. **Agent Expansion**:
   Add more specialized cores for tasks like coding, research, and creative projects.

3. **Vision Integration**:
   Incorporate vision capabilities using LLaVA for object recognition and environmental awareness.

4. **Mobile Integration**:
   Develop dedicated Android and iOS apps for seamless interaction.

---

## Acknowledgments
A heartfelt thanks to the creators of the original **GLaDOS Personality Core** project and its amazing community. Your work has been an inspiration and a foundation for L.U.M.I.N.A.'s development. Join their [Discord community](https://discord.com/invite/ERTDKwpjNB) to learn more and connect with like-minded enthusiasts.



Thank you for exploring L.U.M.I.N.A. Together, let's push the boundaries of AI innovation!




## Software Architecture
The initial goals are to develop a low-latency platform, where GLaDOS can respond to voice interactions within 600ms.

To do this, the system constantly records data to a circular buffer, waiting for [voice to be detected](https://github.com/snakers4/silero-vad). When it's determined that the voice has stopped (including detection of normal pauses), it will be [transcribed quickly](https://github.com/huggingface/distil-whisper). This is then passed to streaming [local Large Language Model](https://github.com/ggerganov/llama.cpp), where the streamed text is broken by sentence, and passed to a [text-to-speech system](https://github.com/rhasspy/piper). This means further sentences can be generated while the current is playing, reducing latency substantially.

### Subgoals
 - The other aim of the project is to minimize dependencies, so this can run on constrained hardware. That means no PyTorch or other large packages.
 - As I want to fully understand the system, I have removed a large amount of redirection: which means extracting and rewriting code. i.e. as GLaDOS only speaks English, I have rewritten the wrapper around [espeak](https://espeak.sourceforge.net/) and the entire Text-to-Speech subsystem is about 500 LOC and has only 3 dependencies: numpy, onnxruntime, and sounddevice.

## Hardware System
This will be based on servo- and stepper-motors. 3D printable STL will be provided to create GlaDOS's body, and she will be given a set of animations to express herself. The vision system will allow her to track and turn toward people and things of interest.

## Installation Instruction


### *New Simplified  Windows Installation Process*
Don't want to compile anything?  Try this simplified process, but be aware it's still in the experimental stage!


1. Open the Microsoft Store, search for `python` and install Python 3.12.
   a. To use Python 3.10, install `typing_extensions` and replace `import typing` in `glados/llama.py` with `import typing_extensions`.
2. Download and unzip this repository somewhere in your home folder.
3. Run the `install_windows.bat`. During the process, you will be prompted to install eSpeak-ng, which is necessary for GLaDOS's speech capabilities. This step also downloads the Whisper voice recognition model and the Llama-3 8B model.
4. Once this is all done, you can initiate  GLaDOS with the `start_windows.bat` script.

### *Even newer Simplified macOS Installation Process*
This is still experimental. Any issues can be addressed in the Discord server. If you create an issue related to this, you will be referred to the Discord server.


1. Install Python 3.12 from pythons website (https://www.python.org/downloads/release/python-3124/)
2. (Optional) Install Homebrew before running the `install_mac.sh`. If you don't do this, it will install it for you (Not tested).
3. `git clone` this repository using `git clone github.com/dnhkng/glados.git`
4. Run the `install_mac.sh`. If you do not have Python installed, then you will run into an error.
5. Once this finishes run the `start_mac.sh` to start GLaDOS

## Regular installation

If you want to install the TTS Engine on your machine, please follow the steps
below.  This has only been tested on Linux, but I think it will work on Windows with small tweaks.
If you are on Windows, I would recommend WSL with an Ubuntu image.  Proper Windows and Mac support is in development.

1. Install the [`espeak`](https://github.com/espeak-ng/espeak-ng) synthesizer
   according to the [installation
   instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)
   for your operating system.
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt` on Mac or Linux systems without an Nvidia GPU, and `pip install -r
   requirements_cuda.txt` if you have a modern Nvidia GPU.
3.  Download the models:
    1.  [voice recognition model](https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin?download=true)
    2.  [Llama-3 8B](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf?download=true) or
    3.  [Llama-3 70B](https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-IQ4_XS.gguf?download=true)
    and put them in the ".models" directory.
4. For voice recognition, we use [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
   1. You can either download the compiled [whisper.cpp DLLs](https://github.com/ggerganov/whisper.cpp/releases) (recommended for Windows), and copy the dll to the ./submodules/whisper.cpp directory
   2. Or compile them yourself.
      1. To pull the code, from the GLaDOS directory use: `git submodule update --init --recursive`
      2. Move to the right subdirectory: `cd submodules/whisper.cpp`
      3. Compile for your system [(see the Documentation)](https://github.com/ggerganov/whisper.cpp), e.g.
         1. Linux with [CUDA](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#nvidia-gpu-support): `WHISPER_CUDA=1 make libwhisper.so -j`
         2. Mac: `make libwhisper.so -j`. For Apple silicon devices, it is also possible to compile using Core ML like this `WHISPER_COREML=1 make libwhisper.so -j`,
            but it may be unnecessary--modern Macs are fast enough without it--and if you do, don't forget to follow the [instructions](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#core-ml-support) to generate Core ML models.
5. For the LLM, you have two options:
   1. Compile llama.cpp:
      1. Use: `git submodule update --init --recursive` to pull the llama.cpp repo
      2. Move to the right subdirectory: `cd submodules/llama.cpp`
      3. Compile llama.cpp, [(see the Documentation)](https://github.com/ggerganov/whisper.cpp)
         1. Linux with [CUDA](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#cuda) `make llama-server LLAMA_CUDA=1`
         2. MacOS with [Metal](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#metal-build) `make llama-server`
   2. Use a commercial API or install an inference backend yourself, such as Ollama or Llamafile:
      1. Find and install a backend with an OpenAI compatible API (most of them)
      2. Edit the glados_config.yaml
         1. update `completion_url` to the URL of your local server
         2. for commercial APIs, add the `api_key`
         3. remove the LlamaServer configurations (make them null)


## Help Section

1. If you have an error about packages or files not being found, make sure you have the whisper and llama binaries in the respective submodules folders!  They are empty by default, and you manually have to add the binaries as described above!

2. Make sure you are using the right Llama-3 Model! I have made Llama-3 8B, with the quantization Q6_K the default. You might need to redownload the model if you don't have `Meta-Llama-3-8B-Instruct-Q6_K.gguf` in your models folder!

3. If you have limited VRAM, you can save 3Gb by using downloading a [highly quantised IQ3_XS model](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-IQ3_XS.gguf?download=true) and moving it to the models folder. If you do this, modify the `glados_config.yaml` to modify the model used: `model_path: "./models/Meta-Llama-3-8B-Instruct-IQ3_XS.gguf"`

4. If you find you are getting stuck in loops, as GLaDOS is hearing herself speak, you have two options:
   1. Solve this by upgrading your hardware. You need to you either headphone, so she can't physically hear herself speak, or a conference-style room microphone/speaker. These have hardware sound cancellation, and prevent these loops.
   2. Disable voice interruption. This means neither you nor GLaDOS can interrupt when GLaDOS is speaking. To accomplish this, edit the `glados_config.yaml`, and change `interruptible:` to  `false`.
