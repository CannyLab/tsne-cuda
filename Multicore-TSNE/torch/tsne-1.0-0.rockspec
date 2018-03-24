package = "tsne"
version = "1.0-0"

source = {
   url = "git://github.com/DmitryUlyanov/Multicore-TSNE.git",
   tag = "master"
}

description = {
   detailed = [[ Wrapper for multicore t-SNE implementation. ]],
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
mkdir -p multicore_tsne/release ; rm -r multicore_tsne/release/* ; cd multicore_tsne/release ; cmake -DCMAKE_BUILD_TYPE=RELEASE .. ; make VERBOSE=1
cd ../..
cp -f multicore_tsne/release/libtsne_multicore.so torch
cd torch
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd torch/build && $(MAKE) install"
}
