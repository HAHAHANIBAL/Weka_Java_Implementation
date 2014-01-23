import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;



public class UntunedLAD {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/spambase_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/spambase_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);


        SMO vs=new SMO();


        vs.buildClassifier(train);
        //find optimal parameter
        //ps.addCVParameter("F 1 5 10");
        //ps.addCVParameter("S 1 10 10");

        //Dagging cls = new Dagging();
        //change the base classifier
        //cls.setClassifier(new NBTree());
        //change the parameter for dagging
        //cls.setNumFolds(1);
        //cls.setSeed(7);
        //cls.buildClassifier(train);
        //System.out.println(vs.getCombinationRule());
        //System.out.println(vs.getOptions());
        PrintWriter pw=new PrintWriter(new FileWriter("/Weka-3-6/ProjectMilestone5/spambase-LB.txt"));

        //System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = vs.classifyInstance(test.instance(i));
            pw.println(pred);
        }
        pw.close();
        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone3/ionosphere.model", vs);






    }
}