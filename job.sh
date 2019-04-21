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
