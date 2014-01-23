import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.LogitBoost;
import weka.core.Instances;
import weka.core.Utils;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;


public class TuningDagging {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/mfeat-factors_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/mfeat-factors_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        CVParameterSelection ps = new CVParameterSelection();
        ps.setClassifier(new Dagging());
        //find optimal parameter

        ps.setOptions(weka.core.Utils.splitOptions("-P \"F 1.0 10.0 1.0\" -P \"S 1.0 10.0 1.0\" -W \"weka.classifiers.meta.Dagging\" -- -W \"weka.classifiers.trees.LMT\" -- -I 5 -A"));

        Dagging cls = new Dagging();
        //change the base classifier


        ps.buildClassifier(train);
        System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        String[] options=ps.getBestClassifierOptions();
        //change the parameter for dagging
        cls.setOptions(options);

        cls.buildClassifier(train);
        PrintWriter pw=new PrintWriter(new FileWriter("/Weka-3-6/ProjectMilestone5/mfeat-factors-L5.txt"));


        for (int i = 0; i < test.numInstances(); i++) {
            double pred = cls.classifyInstance(test.instance(i));
            pw.println(pred);
        }
        pw.close();
        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone2/hypothyroid21.model", cls);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(cls,test);
        Double error_c=eval.errorRate();
        System.out.println(error_c);





    }
}