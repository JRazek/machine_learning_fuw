FROM archlinux:latest as machine_learning_fuw
WORKDIR /root/
RUN pacman -Syu --noconfirm rustup gcc git
RUN git clone https://github.com/JRazek/machine_learning_fuw.git
RUN rustup default nightly
RUN cd machine_learning_fuw && cargo build --release

FROM machine_learning_fuw as poly_fit
WORKDIR /root/machine_learning_fuw
ENTRYPOINT cargo run --release --bin poly_fit
