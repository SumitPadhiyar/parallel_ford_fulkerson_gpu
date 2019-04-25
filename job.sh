cd /home/cs18s042/project
for i in 1 2 3
do
    ./a.out ./dataset/50v.in 50 >> 50.out
    ./a.out ./dataset/100v.in 100 >> 100.out
    ./a.out ./dataset/500v.in 500 >> 500.out
    ./a.out ./dataset/750v.in 750 >> 750.out
    ./a.out ./dataset/1000v.in 1000 >> 1k.out
    ./a.out ./dataset/2000v.in 2000 >> 2k.out
    ./a.out ./dataset/3000v.in 3000 >> 3k.out
    ./a.out ./dataset/4000v.in 4000 >> 4k.out
    ./a.out ./dataset/5000v.in 5000 >> 5k.out
    #./a.out ./dataset/10000v.in 10000 >> 10k.out
done

/usr/local/cuda/bin/nvprof --unified-memory-profiling off ./a.out ./dataset/10000v.in 10000

cd /home/cs18s042/project
for i in 1 2 3 4 5
do
    ./a.out ./dataset/50v.in 50 >> out
    ./a.out ./dataset/100v.in 100 >> out
    ./a.out ./dataset/500v.in 500 >> out
    ./a.out ./dataset/750v.in 750 >> out
    ./a.out ./dataset/1000v.in 1000 >> out
    ./a.out ./dataset/2000v.in 2000 >> out
done
