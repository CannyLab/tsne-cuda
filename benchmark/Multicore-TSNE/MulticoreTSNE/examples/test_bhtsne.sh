# build https://github.com/lvdmaaten/bhtsne first
# download mnist and dump it then
# np.savetxt('/home/dulyanov/mnist.txt', mnist, delimiter='\t')
# cmd to run:
cat /home/dulyanov/mnist.txt | python bhtsne.py --no_pca --perplexity 30 --verbose