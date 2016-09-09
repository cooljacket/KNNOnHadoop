import java.io.IOException;
import java.util.StringTokenizer;
import java.io.DataInput;
import java.io.DataOutput;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Vector;
import java.util.Map;
import java.util.HashMap;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;



public class KNN {
	//public static Vector<Vector<BigDecimal>> test_x = new Vector<>();
   // public static Vector<String> test_y = new Vector<>();
    public static int KNN_K = 3;


	public static class Elem implements WritableComparable {
		private Text dist;
		private Text label;

		public Elem(double dist, String label) {
			this.dist = new Text(""+dist);
			this.label = new Text(label);
		}

		public Elem() {
			dist = new Text();
			label = new Text();
		}

		public void readFields(DataInput in) throws IOException {
			dist.readFields(in);
			label.readFields(in);
		}

		public void write(DataOutput out) throws IOException {
			dist.write(out);
			label.write(out);
		}

		public Double getDist() {
			return Double.parseDouble(dist.toString());
		}

		public String getLabel() {
			return label.toString();
		}

		public int compareTo(Object o) {
			Elem e = (Elem)o;
			if (!this.getDist().equals(e.getDist()))
				return this.getDist().compareTo(e.getDist());
			if (!this.getLabel().equals(e.getLabel()))
				return this.getLabel().compareTo(e.getLabel());
			return 0;
		}
	}


	public static class KNNMapper extends Mapper<LongWritable, Text, IntWritable, Elem> {
		private IntWritable one = new IntWritable(1);
		public static Vector<Vector<BigDecimal>> test_x = new Vector<>();
	    public static Vector<String> test_y = new Vector<>();
	
		protected void setup(Context context) throws IOException, InterruptedException {
			Vector<String> testFile = new Vector<>();
			testFile.add("5.1,3.5,1.4,0.2,Iris-setosa");
			testFile.add("4.6,3.1,1.5,0.2,Iris-setosa");
			testFile.add("4.7,3.2,1.3,0.2,Iris-setosa");
			testFile.add("4.9,3.0,1.4,0.2,Iris-setosa");
			
			for (int i = 0; i < testFile.size(); ++i) {
        	    Vector<BigDecimal> x = new Vector<>();
        	    String y = SplitData(testFile.get(i), x);
        	    test_y.add(y);
        	    Min_Max_Norm(x);
        	    test_x.add(x);
        	}

			System.out.println("[setup-map] " + ", " + test_x.size() + ", " + test_y.size());
		}

		public void map(LongWritable lineOffset, Text line, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(line.toString());
			Vector<BigDecimal> train_x = new Vector<>();
			String train_y = "";

			while (itr.hasMoreTokens()) {
				// the final one is the label
				train_y = itr.nextToken();
				if (itr.hasMoreElements())
					train_x.add(new BigDecimal(train_y));
			}

			if (train_y.equals(""))
				return ;
			
			System.out.println("mapper: " + lineOffset + ", " + line + ", " + train_y + ", " + test_x.size()); 
			

			Min_Max_Norm(train_x);
			for (int i = 0; i < test_x.size(); ++i) {
				IntWritable x = new IntWritable(i);
				context.write(x, new Elem(EulerDistance(train_x, test_x.get(i)), train_y));
				System.out.println("[context.write: ]" + i + ", " + train_y + ", " + EulerDistance(train_x, test_x.get(i)));
			}
		}
	}


	public static class KNNCombiner extends Reducer<IntWritable, Elem, IntWritable, Elem> {
		public void reduce(IntWritable id, Iterable<Elem> values, Context context) throws IOException, InterruptedException {
			TopKElem tk = new TopKElem(KNN_K);
			for (Elem v : values) {
				tk.insert(v);
			}

			Elem[] tops = tk.getTopK();
			for (int i = 0; i < tops.length; ++i) {
				context.write(id, tops[i]);
				System.out.println("[Combiner] " + i + ", " + id + ", " + tops[i].getDist() + ", " + tops[i].getLabel());
			}
		}
	}

	
	public static class KNNReducer extends Reducer<IntWritable, Elem, Text, Text> {
		public void reduce(IntWritable id, Iterable<Elem> values, Context context) throws IOException, InterruptedException {
			System.out.println("Befor reduce " + id + ", " + KNNMapper.test_x.size() + ", " + KNNMapper.test_y.size());
			Text actual_label = new Text("actual");//new Text(KNNMapper.test_y.get(id.get()));

			TopKElem tk = new TopKElem(KNN_K);
			for (Elem v : values) {
				tk.insert(v);
			}

			Elem[] tops = tk.getTopK();
			Map<String, Integer> c_map = new HashMap<String, Integer>();
			for (int i = 0; i < tops.length; ++i) {
				String key = tops[i].getLabel();
				int cnt = 0;
				if (c_map.get(key) != null)
					cnt = c_map.get(key);
				c_map.put(key, cnt+1);

				System.out.println("[Reducer] " + i + ", " + key + ", " + (cnt+1));
			}

			Integer maxN = 0;
	        String predict_label = "";
	        for (Map.Entry<String, Integer> entry : c_map.entrySet()) {
	            Integer v = entry.getValue();
	            if (v.compareTo(maxN) > 0) {
	                maxN = v;
	                predict_label = entry.getKey();
	            }
	        }

	        context.write(new Text(actual_label), new Text(predict_label));
			System.out.println("Final " + actual_label + ", " + predict_label);
		}
	}


	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();		
		args = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (args.length != 2) {
			System.err.println("Usage: KNN <input> <output>");
			System.exit(2);
		}

		String trainFile = args[0];
		String outputDirName = args[1];



		Job job = new Job(conf, "KNN");
		job.setJarByClass(KNN.class);
		job.setMapperClass(KNNMapper.class);
		job.setCombinerClass(KNNCombiner.class);
		// job.setPartitionerClass(KNNPartitioner.class);
		job.setReducerClass(KNNReducer.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Elem.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(trainFile));
		FileOutputFormat.setOutputPath(job, new Path(outputDirName));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}


	public static String SplitData(String raw, Vector<BigDecimal> X) {
        String[] data = raw.split(",");
        for (int i = 0; i < data.length-1; ++i)
            X.add(new BigDecimal(data[i]));
        return data[data.length-1];
    }


    public static void Min_Max_Norm(Vector<BigDecimal> X) {
        BigDecimal small = X.get(0), big = X.get(0);
        for (int i = 0; i < X.size(); ++i) {
            BigDecimal now = X.get(i);
            if (small.compareTo(now) > 0)
                small = now;
            if (big.compareTo(now) < 0)
                big = now;
        }

        BigDecimal interval = big.subtract(small);
        for (int i = 0; i < X.size(); ++i) {
            X.set(i, (X.get(i).subtract(small)).divide(interval, MathContext.DECIMAL64));
        }
    }


    public static double EulerDistance(Vector<BigDecimal> x, Vector<BigDecimal> y) {
        double dist = 0.0, tmp;
        int size = x.size();
        for (int i = 0; i < size; ++i) {
            tmp = (x.get(i).subtract(y.get(i))).doubleValue();
            dist += tmp * tmp;
        }

        return Math.sqrt(dist);
    }


    public static class TopKElem {
        private Elem[] topK;
        private int index, size;

        public TopKElem(int k) {
            index = -1;
            size = k;
            topK = new Elem[size+1];
        }

        public void insert(Elem x) {
            index = Math.min(index+1, size);
            int pos = index - 1;

            while (pos >= 0 && x.getDist().compareTo(topK[pos].getDist()) < 0) {
                topK[pos+1] = topK[pos];
                --pos;
            }
            if (pos < size)
                topK[pos+1] = x;
        }

        public Elem[] getTopK() {
            if (index < 0)
                return null;
            int num = Math.min(index+1, size);
            Elem[] data = new Elem[num];
            for (int i = 0; i < num; ++i)
                data[i] = topK[i];
            return data;
        }
    }
}