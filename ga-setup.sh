#=================================================
#=GA=Settings
#=================================================

export TARGET=LINUX64
export USE_MPI=yes
export ARMCI_NETWORK=OPENIB

MPI_DIR=/soft/libraries/mpi/mvapich2/gcc
MPICH_LIBS="-lmpich -lopa -lmpl -lpthread -lrdmacm -libverbs -libumad -ldl -lrt -lnuma"
export MPI_LIB="${MPI_DIR}/lib"
export MPI_INCLUDE="${MPI_DIR}/include"
export LIBMPI="-L${MPI_DIR}/lib -Wl,-rpath -Wl,${MPI_DIR}/lib ${MPICH_LIBS}"

export GA_CONFIG=/home/pghosh/opt/ga_5.1.1-install/bin/ga-config
#=================================================
#=IB SETTINGS
#=================================================

export IB_HOME=/usr
export IB_INCLUDE=$IB_HOME/include
export IB_LIB=$IB_HOME/lib64

