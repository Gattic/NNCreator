mkdir -p build

ENABLE_SDL2=OFF
ENABLE_OPENGL=ON

for arg in "$@"; do
	case "$arg" in
		sdl=on|--sdl=on)
			ENABLE_SDL2=ON
			;;
		sdl=off|--sdl=off)
			ENABLE_SDL2=OFF
			;;
		opengl=on|--opengl=on)
			ENABLE_OPENGL=ON
			;;
		opengl=off|--opengl=off)
			ENABLE_OPENGL=OFF
			;;
		both|--both)
			ENABLE_SDL2=ON
			ENABLE_OPENGL=ON
			;;
		sdl|--sdl)
			ENABLE_SDL2=ON
			ENABLE_OPENGL=OFF
			;;
		opengl|--opengl)
			ENABLE_SDL2=OFF
			ENABLE_OPENGL=ON
			;;
		*)
			;;
	esac
done

cd build
cmake -DENABLE_SDL2=${ENABLE_SDL2} -DENABLE_OPENGL=${ENABLE_OPENGL} ..
make -j$(nproc)
