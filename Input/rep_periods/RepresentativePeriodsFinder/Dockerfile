FROM julia:latest

WORKDIR /home/

COPY . .

# RUN apt-get -qq update; apt-get -y --no-install-recommends install git qt5-default at-spi2-core libgtk-3-dev xauth xvfb gconf-gsettings-backend

# qt5-default is not available for Ubuntu 21.04, hence the modification to the line below

RUN apt-get -qq update; apt-get -y --no-install-recommends install git qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools at-spi2-core libgtk-3-dev xauth xvfb gconf-gsettings-backend

RUN export GKS_WSTYPE=100

RUN DIR="./TimeSeriesClustering.jl/"; if ! [ -d $DIR ] ; then git clone https://github.com/junglegobs/TimeSeriesClustering.jl.git $DIR ; fi

RUN julia -E ' \
            ENV["GKSwstype"]="100"; \
            using Pkg; \
            Pkg.activate("."); \
            Pkg.instantiate(); \
            Pkg.precompile(); \
            '

RUN echo 'using Pkg; Pkg.activate(".")' > /usr/local/julia/etc/julia/startup.jl 

CMD ["julia"]
