n=13;#the variable that I want to be incremented
next_n=$[$n+1]
sed -i "/#the variable that I want to be incremented$/s/=.*#/=$next_n;#/" ${0}
echo $n

mkdir results/$n

python3 extract-features.py ../../../data/Train/DrugBank > results/$n/train.cod
python3 train-naive.py results/$n/NaiveBayes.pickle < results/$n/train.cod

# encode test corpus into features and apply learned model
python3 extract-features.py ../../../data/Test-NER/DrugBank > results/$n/test.cod
python3 predict-naive.py results/$n/NaiveBayes.pickle < results/$n/test.cod  > results/$n/task9.1.txt

# evaluate results with official scorer
java -jar ../../../eval/evaluateNER.jar ../../../data/Test-NER/DrugBank results/$n/task9.1.txt

echo $n >> results/summary.txt
tail -5 results/$n/task9.1_scores.log >> results/summary.txt
