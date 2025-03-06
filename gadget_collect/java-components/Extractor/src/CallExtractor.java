import soot.*;
import java.util.*;

public class CallExtractor implements Extractor{

    private Soot_utlilty utility = new Soot_utlilty();

    @Override
    public SootClass extract_feature(String feature,String name_folder) {
        SootClass slice_class = null;
        ArrayList<SootClass> entrypoints = utility.extract_class_call(feature);
        for (Iterator<SootClass> iter = entrypoints.iterator(); iter.hasNext(); ) {
            SootClass entrypoint = iter.next();
            ArrayList<SootMethod> method_entrypoints = utility.find_method_for_feature(entrypoint, feature);
            //Check if we need extra dependencies
            ArrayList<String> dependencies_tot = new ArrayList<>();
            // usualy it does not need any dependency, check if android is in the library name
            ArrayList<String> dependencies = utility.extract_activity_dependencies_PDG(new ArrayList<String>(), entrypoint.getName());
            dependencies_tot.addAll(dependencies);
            boolean done = false;
            Iterator<SootMethod> target_iterator = method_entrypoints.iterator();
            while (done == false && target_iterator.hasNext()) {
                SootMethod method = target_iterator.next();
                SootMethod corresponding_sootMethod = utility.find_SootMethodNoClass(feature,method);
                if (corresponding_sootMethod != null){
                    ArrayList<My_slice> slices = utility.extract_method_call_method(entrypoint, method, corresponding_sootMethod);
                    if (!slices.isEmpty()) {
                        My_slice simplest = null;
                        if (slices.size() > 1) {
                            simplest = utility.get_simpler_slice(slices);
                        } else {
                            simplest = slices.get(0);
                        }
                        simplest.setName("Slice" + name_folder + "interesting_call");
                        simplest.setFeature("tmp");
                        SootClass tmp = simplest.get_soot_class();
                        System.out.println(simplest.toString());
                        utility.WriteFile("class_of_extraction.txt", entrypoint.getName());
                        slice_class = tmp;
                        done = true;
                    } else {
                        System.out.println("No possible slice found :(");
                    }
                }
            }
            if (!dependencies_tot.isEmpty()) {
                Set<String> foo = new HashSet<String>(dependencies_tot);
                ArrayList<String> mainList = new ArrayList<String>();
                mainList.addAll(foo);
                utility.extract_classes(mainList);
            }
            if (done){
                break;
            }
        }
        return slice_class;
    }
}
