import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;



public class SMOwithFilter {

    public static void main(String[] args) throws Exception{

        // load data sets
        Instances train = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/sonar_train.arff")));
        Instances test = new Instances(
                new BufferedReader(
                        new FileReader("/Weka-3-6/ProjectMilestone5/sonar_test.arff")));
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes()-1);
        NonSparseToSparse filter = new NonSparseToSparse();
        filter.setInputFormat(train);
        Instances newtrain = Filter.useFilter(train, filter);
        Instances newtest = Filter.useFilter(test, filter);
        newtrain.setClassIndex(newtrain.numAttributes() - 1);
        newtest.setClassIndex(newtest.numAttributes()-1);
        //Classifier [] ClassifierArray=new Classifier[3];
        //ClassifierArray[1]=new J48();
        //ClassifierArray[0]=new NaiveBayes();
        //ClassifierArray[2]=new NBTree();
        SMO vs = new SMO();

        //find optimal parameter

        vs.setOptions(weka.core.Utils.splitOptions(""));
        //String[] options=new String[3];
        //options[2]="-R MAJ";
        //options[1]="-B weka.classifiers.functions.SMO -B weka.classifiers.bayes.NaiveBayes";
        //options[0]="-S <2>";
        //vs.setOptions(options);

        //vs.setClassifiers(ClassifierArray);

        vs.buildClassifier(newtrain);
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
        PrintWriter pw=new PrintWriter(new FileWriter("/Weka-3-6/ProjectMilestone5/sonar-L5.txt"));

        //System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
        for (int i = 0; i < newtest.numInstances(); i++) {
            double pred = vs.classifyInstance(newtest.instance(i));
            pw.println(pred);
        }
        pw.close();
        //weka.core.SerializationHelper.write("/Weka-3-6/ProjectMilestone3/ionosphere.model", vs);





    }
}