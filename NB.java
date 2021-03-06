import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;



public class NB {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone4/diabetes_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone4/diabetes_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);

        //Classifier [] ClassifierArray=new Classifier[3];
        //ClassifierArray[1]=new J48();
        //ClassifierArray[0]=new NaiveBayes();
        //ClassifierArray[2]=new NBTree();
        NaiveBayes vs=new NaiveBayes();
        //String[] options=new String[3];
        //options[2]="-R MAJ";
        //options[1]="-B weka.classifiers.functions.SMO -B weka.classifiers.bayes.NaiveBayes";
        //options[0]="-S <2>";
        //vs.setOptions(options);

        //vs.setClassifiers(ClassifierArray);

        vs.buildClassifier(train);
        //find optimal parameter
        //ps.addCVParameter("F 1 5 5");
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
        //PrintWriter pw=new PrintWriter(new FileWriter("/Weka-3-6/ProjectMilestone4/balance-scale1.txt"));

        //System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        //for (int i = 0; i < test.numInstances(); i++) {
        //    double pred = vs.classifyInstance(test.instance(i));
        //    pw.println(pred);
        //}
        //pw.close();
        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone3/ionosphere.model", vs);
        Evaluation eval=new Evaluation(train);
        eval.evaluateModel(vs,test);
        Double error_c=eval.errorRate();
        System.out.println(error_c);





    }
}