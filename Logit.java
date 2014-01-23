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


public class Logit {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/arrhythmia_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/arrhythmia_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        CVParameterSelection ps = new CVParameterSelection();
        ps.setClassifier(new LogitBoost());
        //find optimal parameter

        ps.setOptions(weka.core.Utils.splitOptions("-P \"P 10 100 10\" -P \"F 1 5 1\" -P \"R 1 5 1\" -P \"I 1 20 1\" -P \"H 1 5 1\" -W \"weka.classifiers.meta.LogitBoost\" -- -W \"weka.classifiers.functions.LinearRegression\""));

        LogitBoost cls = new LogitBoost();
        //change the base classifier


        ps.buildClassifier(train);
        System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        String[] options=ps.getBestClassifierOptions();
        //change the parameter for dagging
        cls.setOptions(options);

        cls.buildClassifier(train);
        PrintWriter pw=new PrintWriter(new FileWriter("/Weka-3-6/ProjectMilestone5/arrhythmia-L5.txt"));


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