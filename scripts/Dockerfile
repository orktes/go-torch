FROM debian:9.7

ENV DEBIAN_FRONTEND=noninteractive
ENV GO_VERSION 1.13

RUN apt-get update && \
  apt-get install -y g++ pkg-config ca-certificates unzip wget && \
  apt-get clean && \
  rm -fr /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cd /opt/ && \
  wget -q https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.1.zip && \
  unzip libtorch-cxx11-abi-shared-with-deps-1.3.1.zip && \
  rm libtorch-cxx11-abi-shared-with-deps-1.3.1.zip

RUN echo "$PYTORCH_DIST_DIR/lib" >> /etc/ld.so.conf.d/libtorch.conf && ldconfig
ENV LD_LIBRARY_PATH /opt/libtorch/lib:${LD_LIBRARY_PATH}

RUN  wget -qO- https://dl.google.com/go/go${GO_VERSION}.linux-amd64.tar.gz \
  | tar -C /usr/local -xz && \
  export PATH="/usr/local/go/bin:$PATH" && \
  go version;

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"



