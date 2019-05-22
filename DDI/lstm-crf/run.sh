n=4;#the variable that I want to be incremented
next_n=$[$n+1]
sed -i "/#the variable that I want to be incremented$/s/=.*#/=$next_n;#/" ${0}
echo $n

mkdir results/$n

python3 extract-features.py ../../../data/Train/DrugBank > results/$n/train.cod

# encode test corpus into features and apply learned model
python3 extract-features.py ../../../data/Test-DDI/DrugBank > results/$n/test.cod
python3 train-lstm.py results/$n/train.cod results/$n/test.cod results/$n/task9.1.txt
#python3 train-lstm.py results/1/train.cod results/1/test.cod results/1/task9.1.txt

# evaluate results with official scorer
java -jar ../../../eval/evaluateDDI.jar ../../../data/Test-DDI/DrugBank results/1/task9.1.txt

echo $n >> results/summary.txt
tail -5 results/$n/task9.1_scores.log >> results/summary.txt



#python3 train-lstm.py results/0/model.pickle results/0/word2idx results/0/utags results/0/train.cod results/0/test.cod results/0/task9.1.txt
#python3 predict-lstm.py results/0/model.pickle results/0/word2idx results/0/utags < results/0/test.cod  > results/0/task9.1.txt
