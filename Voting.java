import weka.classifiers.Evaluation;
import weka.classifiers.meta.GridSearch;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.Utils;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;



public class Voting {

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

        //Classifier[] ClassifierArray=new Classifier[3];
        //ClassifierArray[1]=new J48();
        //ClassifierArray[0]=new NaiveBayes();
        //ClassifierArray[2]=new LogitBoost();
        Vote vs=new Vote();
        //String[] options=new String[3];
        //options[2]="-R MAJ";
        //options[1]="-B weka.classifiers.functions.SMO -B weka.classifiers.bayes.NaiveBayes";
        //options[0]="-S <2>";
        //vs.setOptions(options);
        GridSearch ps = new GridSearch();
        //ps.setOptions(weka.core.Utils.splitOptions("-P \"I 1.0 10.0 1.0\" -P \"P 1.0 100.0 10.0\" -W \"weka.classifiers.meta.AdaBoostM1\" -- -P 100 -S 1 -I 10 -W \"weka.classifiers.functions.SMO\" -- -V -1 -C 1 -P 1.0E-12"));
        vs.setOptions(weka.core.Utils.splitOptions("-B \"weka.classifiers.functions.SMO -C 1 -L 0.01 -P 1E-10\" -B \"weka.classifiers.trees.NBTree\" -B \"weka.classifiers.trees.RandomForest -I 10 -K 20 -depth 5\" -R MAJ"));

        //String[] options=ps.getBestClassifierOptions();

        //vs.setOptions(options);
        System.out.println(Utils.joinOptions(vs.getOptions()));
        //vs.setClassifiers(ClassifierArray);

        vs.buildClassifier(train);
        //find optimal parameter



        //Dagging cls = new Dagging();
        //change the base classifier
        //cls.setClassifier(new NBTree());
        //change the parameter for dagging
        //cls.setNumFolds(1);
        //cls.setSeed(7);
        //cls.buildClassifier(train);
        //System.out.println(vs.getCombinationRule());
        //System.out.println(vs.getOptions());
        PrintWriter pw=new PrintWriter(new FileWriter("/Weka-3-6/ProjectMilestone5/spambase-L5.txt"));

        //System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = vs.classifyInstance(test.instance(i));
            pw.println(pred);
        }
        pw.close();
        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone3/ionosphere.model", vs);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(vs,test);
        Double error_c=eval.errorRate();
        System.out.println(error_c);





    }
}