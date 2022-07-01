# Install, Compile, and Run

---

## Dependencies

`cmake`

`make`

`g++`

SDL2 graphics library
`libsdl2-dev`

SDL2 TTF for GUI
`libsdl2-ttf-dev`

SDL2 Image for GUI
`libsdl2-image-dev`

OpenSSL
`libssl-dev`
`libcurl4-openssl-dev`

Optional:
`clang-format`
`gdb`

One-liner for convenience:
```
sudo apt-get install cmake make gcc g++ libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev libssl-dev clang-format libcurl4-openssl-dev gdb
```

## Python Dependencies
```
https://github.com/mtusman/gemini-python
```

---

## Compilation

```
mkdir build
cd build
cmake ..
make
```

---

## Installation

make install

---

## Uninstall

make uninstall

## Run

```
cd build
make run
```

### Run Debugging (gdb)

```
cd build
make debug
backtrace # When it crashes
```

### Run Fullscreen

```
cd build
make fullscreen
```

### Run Terminal Only

```
cd build
make nogui
```

---

## Method for Running Native on Windows 10 (without Cygwin)

**Step 1) Turn on Windows Subsystem for Linux**
1. Run PowerShell as administrator.
2. Enter `Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux`
3. Restart your computer if prompted.

**Step 2) Get Ubuntu on Windows**
1. Open Microsoft Store
2. In the search bar, search 'Ubuntu'
3. Select the app 'Ubuntu'
4. Click 'Get' and then click 'Install'
5. You may need to recover your Microsoft account

**Step 3) Update and upgrade Ubuntu**
1. Run Ubuntu
2. Enter commands: `sudo apt-get update` then `sudo apt-get upgrade`

**Step 4) Get XMing**
1. Go to [this link](https://sourceforge.net/projects/xming/)
2. Download and install XMing somewhere you can find it
3. Open WSL Bash Terminal
4. `nano ~/.bashrc #or vim/emacs or whatever`
5. Add to bottom `export DISPLAY=:0`
6. Save and exit nano/vim/emacs or whatever
7. Exit/Restart terminal

**Step 5) Get NNCreator**
1. Install all dependencies listed for Linux above in the Ubuntu Bash terminal
2. Clone the repository into the directory you want it (within Ubuntu)

**Step 6) Install Sublime on Ubuntu on Windows**
1. This is required for you to edit the same files that you run on Ubuntu
2. Follow the instructions listed [here](https://linuxize.com/post/how-to-install-sublime-text-3-on-ubuntu-18-04/)

**Step 6) Run NNCreator**
1. Run Ubuntu
2. Run XMing
3. In the Ubuntu Bash terminal, type `export DISPLAY=:0`
4. Navigate to NNCreator project and compile and `make run` as usual  
*Note: Step 3 is required each time you open the Ubuntu Bash terminal, and must be performed while XMing is running*
