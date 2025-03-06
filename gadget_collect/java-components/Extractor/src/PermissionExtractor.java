import org.xmlpull.v1.XmlPullParserException;
import soot.SootClass;
import soot.SootMethod;
import soot.jimple.infoflow.android.axml.AXmlNode;
import soot.jimple.infoflow.android.manifest.ProcessManifest;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class PermissionExtractor implements Extractor{
    private Soot_utlilty utility = new Soot_utlilty();


    public void extractPermission(String feature) {
        try {
            ProcessManifest manifest = new ProcessManifest(Instrumenter.apkPath);
            Set<String> permissions = manifest.getPermissions();
            for(String node : permissions){
                if(node.toString().contains(feature)){
                    this.utility.WriteFile("permission.txt",node.toString());
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (XmlPullParserException e) {
            e.printStackTrace();
        }
    }

    private Map<String, ArrayList<String>> parse_SourceSinks(){
        BufferedReader reader,reader2;
        Map<String, ArrayList<String>> to_return= new HashMap<>();
        try {
            reader = new BufferedReader(new FileReader(
                    "/home/entrophy/apg-release/permission_lists/SS.txt"));
            String line = reader.readLine();
            while (line != null) {
                // read next line
                String method = line.split("::")[0].trim();
                String permission = line.split("::")[1].trim();
                if(to_return.containsKey(permission)){
                    ArrayList<String> list_tmp = to_return.get(permission);
                    if (!list_tmp.contains(method)){
                        to_return.get(permission).add(method);
                    }
                }else{
                    ArrayList<String> list_tmp = new ArrayList<>();
                    list_tmp.add(method);
                    to_return.put(permission,list_tmp);
                }
                line = reader.readLine();

            }
            reader.close();
            reader2 = new BufferedReader(new FileReader(
                    "/home/entrophy/apg-release/permission_lists/ics_allmappings"));
            //Permission:android.permission.CHANGE_WIFI_STATE
            line = reader2.readLine();
            String permission = line.split(":")[1].trim();
            //719 Callers:
            reader2.readLine();
            //begins
            line = reader2.readLine();
            while (line != null) {
                if (line.startsWith("Permission:")){
                    permission = line.split(":")[1].trim();
                    reader2.readLine();
                    line = reader2.readLine();
                }
                String method = line.split(":")[1].split(">")[0].trim();
                if(to_return.containsKey(permission)){
                    ArrayList<String> list_tmp = to_return.get(permission);
                    if (!list_tmp.contains(method)){
                        to_return.get(permission).add(method);
                    }
                }else{
                    ArrayList<String> list_tmp = new ArrayList<>();
                    list_tmp.add(method);
                    to_return.put(permission,list_tmp);
                }
                // read next line
                line = reader2.readLine();
            }
            reader2.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return to_return;
    }

    @Override
    public SootClass extract_feature(String feature, String name_folder) {
        SootClass slice_class = null;
        Map<String, ArrayList<String>> map_permission_method  = parse_SourceSinks();
        ArrayList<String> methods_possible = map_permission_method.get(feature);
        for(String methodPossible : methods_possible){
            String methodTmp = methodPossible.split("\\(")[0].split(" ")[1];
            ArrayList<SootClass> entrypoints = utility.extract_class_call(methodTmp);
            for (Iterator<SootClass> iter = entrypoints.iterator(); iter.hasNext(); ) {
                SootClass entrypoint = iter.next();
                ArrayList<SootMethod> method_entrypoints = utility.find_method_for_feature(entrypoint, methodTmp);
                //Check if we need extra dependencies
                ArrayList<String> dependencies_tot = new ArrayList<>();
                // usualy it does not need any dependency, check if android is in the library name
                ArrayList<String> dependencies = utility.extract_activity_dependencies_PDG(new ArrayList<String>(), entrypoint.getName());
                dependencies_tot.addAll(dependencies);
                boolean done = false;
                Iterator<SootMethod> target_iterator = method_entrypoints.iterator();
                while (done == false && target_iterator.hasNext()) {
                    SootMethod method = target_iterator.next();
                    SootMethod corresponding_sootMethod = utility.find_SootMethodNoClass(methodTmp,method);
                    if (corresponding_sootMethod != null){
                        ArrayList<My_slice> slices = utility.extract_method_call_method(entrypoint, method, corresponding_sootMethod);
                        if (!slices.isEmpty()) {
                            My_slice simplest = null;
                            if (slices.size() > 1) {
                                simplest = utility.get_simpler_slice(slices);
                            } else {
                                simplest = slices.get(0);
                            }
                            simplest.setName("Slice" + name_folder + "permission");
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
            }
        }
        return slice_class;
    }
}
