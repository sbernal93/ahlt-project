n=52;#the variable that I want to be incremented
next_n=$[$n+1]
sed -i "/#the variable that I want to be incremented$/s/=.*#/=$next_n;#/" ${0}
echo $n

mkdir results/$n


python3 extract-features.py ../../../data/Train/MedLine > results/$n/train.cod
python3 train-crf.py results/$n/model.crf < results/$n/train.cod


# encode test corpus into features and apply learned model
python3 extract-features.py ../../../data/Test-DDI/MedLine > results/$n/test.cod
python3 predict-crf.py results/$n/model.crf < results/$n/test.cod  > results/$n/task9.1.txt

# evaluate results with official scorer
java -jar ../../../eval/evaluateDDI.jar ../../../data/Test-DDI/MedLine results/$n/task9.1.txt

echo $n >> results/summary.txt
tail -5 results/$n/task9.1_All_scores.log >> results/summary.txt
