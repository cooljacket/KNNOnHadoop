rm *.class *.jar

javac -cp ../share/hadoop/common/hadoop-common-2.6.4.jar:../share/hadoop/mapreduce/hadoop-mapreduce-client-core-2.6.4.jar:../share/hadoop/common/lib/commons-cli-1.2.jar KNN.java -d ./

jar -cvf KNN.jar KNN*.class

hdfs dfs -rm -r KNN_Output

hadoop jar KNN.jar KNN KNN_input/train.data KNN_input/test.data KNN_Output

hdfs dfs -cat KNN_Output/* > output.txt

./out