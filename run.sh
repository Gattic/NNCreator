mkdir -p build

ENABLE_FEDORA=OFF
MAKE_TARGET=run

for arg in "$@"; do
	case "$arg" in
		fedora|--fedora)
			ENABLE_FEDORA=ON
			;;
		debug|--debug)
			MAKE_TARGET=debug
			;;
		nvtx|--nvtx|profile_gpu|--profile_gpu)
			MAKE_TARGET=profile_gpu
			;;
		*)
			;;
	esac
done

cd "build/"

if [[ "${ENABLE_FEDORA}" == "ON" ]]; then
	__NV_PRIME_RENDER_OFFLOAD=1 \
	__GLX_VENDOR_LIBRARY_NAME=nvidia \
	make "${MAKE_TARGET}"
else
	make "${MAKE_TARGET}"
fi
