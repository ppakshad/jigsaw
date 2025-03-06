import org.xmlpull.v1.XmlPullParserException;
import soot.SootClass;
import soot.jimple.infoflow.android.axml.AXmlNode;
import soot.jimple.infoflow.android.manifest.IComponentContainer;
import soot.jimple.infoflow.android.manifest.ProcessManifest;
import soot.jimple.infoflow.android.manifest.binary.BinaryManifestBroadcastReceiver;
import soot.jimple.infoflow.android.manifest.binary.BinaryManifestContentProvider;
import soot.jimple.infoflow.android.manifest.binary.BinaryManifestService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class IntentExtractor implements Extractor{

    private Soot_utlilty utility = new Soot_utlilty();


    private String find_intent(String feature){
        String to_ret = "";
        boolean found = false;
        try {
            ProcessManifest manifest = new ProcessManifest(Instrumenter.apkPath);
            IComponentContainer<BinaryManifestBroadcastReceiver> receivers =  manifest.getBroadcastReceivers();
            IComponentContainer<BinaryManifestContentProvider> providers =  manifest.getContentProviders();
            IComponentContainer<BinaryManifestService> services =  manifest.getServices();
            ArrayList<AXmlNode> activities = manifest.getAllActivities();
            for(BinaryManifestBroadcastReceiver node : receivers){
                List<AXmlNode> childs = node.getAXmlNode().getChildren();
                for (AXmlNode child_node : childs){
                    if(child_node.getTag().equals("intent-filter")){
                        List<AXmlNode> childs_I = child_node.getChildren();
                        for (AXmlNode child_node_I : childs_I){
                            if(child_node_I.hasAttribute("name")){
                                if(child_node_I.getAttribute("name").toString().contains(feature)){
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                    if(found){break;}
                }
                if(found){
                    to_ret = node.getAXmlNode().getTag()+"::"+node.getAXmlNode().getAttribute("name").getValue();
                    break;
                }
            }
            if(!found){
                for(BinaryManifestService node : services){
                    List<AXmlNode> childs = node.getAXmlNode().getChildren();
                    for (AXmlNode child_node : childs){
                        if(child_node.getTag().equals("intent-filter")){
                            List<AXmlNode> childs_I = child_node.getChildren();
                            for (AXmlNode child_node_I : childs_I){
                                if(child_node_I.hasAttribute("name")){
                                    if(child_node_I.getAttribute("name").toString().contains(feature)){
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if(found){break;}
                    }
                    if(found){
                        to_ret = node.getAXmlNode().getTag()+"::"+node.getAXmlNode().getAttribute("name").getValue();
                        break;
                    }
                }
            }
            if(!found){
                for(BinaryManifestContentProvider node : providers){
                    List<AXmlNode> childs = node.getAXmlNode().getChildren();
                    for (AXmlNode child_node : childs){
                        if(child_node.getTag().equals("intent-filter")){
                            List<AXmlNode> childs_I = child_node.getChildren();
                            for (AXmlNode child_node_I : childs_I){
                                if(child_node_I.hasAttribute("name")){
                                    if(child_node_I.getAttribute("name").toString().contains(feature)){
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if(found){break;}
                    }
                    if(found){
                        to_ret = node.getAXmlNode().getTag()+"::"+node.getAXmlNode().getAttribute("name").getValue();
                        break;
                    }
                }
            }
            if(!found){
                for(AXmlNode node : activities){
                    List<AXmlNode> childs = node.getChildren();
                    for (AXmlNode child_node : childs){
                        if(child_node.getTag().equals("intent-filter")){
                            List<AXmlNode> childs_I = child_node.getChildren();
                            for (AXmlNode child_node_I : childs_I){
                                if(child_node_I.hasAttribute("name")){
                                    if(child_node_I.getAttribute("name").toString().contains(feature)){
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if(found){break;}
                    }
                    if(found){
                        to_ret = node.getTag()+"::"+node.getAttribute("name").getValue();
                        break;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (XmlPullParserException e) {
            e.printStackTrace();
        }

        return to_ret;
    }

    @Override
    public SootClass extract_feature(String feature, String name_folder) {
        SootClass slice_class = null;
        String intent_tag = find_intent(feature);
        String type = intent_tag.split("::")[0];
        String name = intent_tag.split("::")[1];
        switch (type){
            case "receiver":
                ReceiverExtractor receiver_extractor = new ReceiverExtractor();
                slice_class = receiver_extractor.extract_feature(name, name_folder);
                break;
            case "provider":
                ProviderExtractor provider_extractor = new ProviderExtractor();
                slice_class = provider_extractor.extract_feature(name, name_folder);
                break;
            case "service":
                ServiceExtractor service_extractor = new ServiceExtractor();
                slice_class = service_extractor.extract_feature(name, name_folder);
                break;
            case "activity":
                Activity_extractor activity_extractor = new Activity_extractor();
                slice_class = activity_extractor.extract_feature(name, name_folder);
                break;
        }
        return slice_class;
    }

    public void extractIntent(String feature) {
        boolean found = false;
        try {
            ProcessManifest manifest = new ProcessManifest(Instrumenter.apkPath);
            IComponentContainer<BinaryManifestBroadcastReceiver> receivers =  manifest.getBroadcastReceivers();
            IComponentContainer<BinaryManifestContentProvider> providers =  manifest.getContentProviders();
            IComponentContainer<BinaryManifestService> services =  manifest.getServices();
            ArrayList<AXmlNode> activities = manifest.getAllActivities();
            for(BinaryManifestBroadcastReceiver node : receivers){
                List<AXmlNode> childs = node.getAXmlNode().getChildren();
                for (AXmlNode child_node : childs){
                    if(child_node.getTag().equals("intent-filter")){
                        List<AXmlNode> childs_I = child_node.getChildren();
                        for (AXmlNode child_node_I : childs_I){
                            if(child_node_I.hasAttribute("name")){
                                if(child_node_I.getAttribute("name").toString().contains(feature)){
                                    found = true;
                                    String tmp_str = "<intent-filter>"+child_node_I.toString();
                                    String new_tmp = tmp_str.substring(0,tmp_str.length()-1).concat("/></intent-filter>");
                                    this.utility.WriteFile("xml_tag.xml",new_tmp);
                                    break;
                                }
                            }
                        }
                    }
                    if(found){break;}
                }
                if(found){
                    this.utility.WriteFile("intent_class.txt",node.getAXmlNode().getAttribute("name").getValue().toString());

                    break;
                }
            }
            if(!found){
                for(BinaryManifestService node : services){
                    List<AXmlNode> childs = node.getAXmlNode().getChildren();
                    for (AXmlNode child_node : childs){
                        if(child_node.getTag().equals("intent-filter")){
                            List<AXmlNode> childs_I = child_node.getChildren();
                            for (AXmlNode child_node_I : childs_I){
                                if(child_node_I.hasAttribute("name")){
                                    if(child_node_I.getAttribute("name").toString().contains(feature)){
                                        found = true;
                                        String tmp_str = "<intent-filter>"+child_node_I.toString();
                                        String new_tmp = tmp_str.substring(0,tmp_str.length()-1).concat("/></intent-filter>");
                                        this.utility.WriteFile("xml_tag.xml",new_tmp);
                                        break;
                                    }
                                }
                            }
                        }
                        if(found){break;}
                    }
                    if(found){
                        this.utility.WriteFile("intent_class.txt",node.getAXmlNode().getAttribute("name").getValue().toString());

                        break;
                    }
                }
            }
            if(!found){
                for(BinaryManifestContentProvider node : providers){
                    List<AXmlNode> childs = node.getAXmlNode().getChildren();
                    for (AXmlNode child_node : childs){
                        if(child_node.getTag().equals("intent-filter")){
                            List<AXmlNode> childs_I = child_node.getChildren();
                            for (AXmlNode child_node_I : childs_I){
                                if(child_node_I.hasAttribute("name")){
                                    if(child_node_I.getAttribute("name").toString().contains(feature)){
                                        found = true;
                                        String tmp_str = "<intent-filter>"+child_node_I.toString();
                                        String new_tmp = tmp_str.substring(0,tmp_str.length()-1).concat("/></intent-filter>");
                                        this.utility.WriteFile("xml_tag.xml",new_tmp);
                                        break;
                                    }
                                }
                            }
                        }
                        if(found){break;}
                    }
                    if(found){
                        this.utility.WriteFile("intent_class.txt",node.getAXmlNode().getAttribute("name").getValue().toString());

                        break;
                    }
                }
            }
            if(!found){
                for(AXmlNode node : activities){
                    List<AXmlNode> childs = node.getChildren();
                    for (AXmlNode child_node : childs){
                        if(child_node.getTag().equals("intent-filter")){
                            List<AXmlNode> childs_I = child_node.getChildren();
                            for (AXmlNode child_node_I : childs_I){
                                if(child_node_I.hasAttribute("name")){
                                    if(child_node_I.getAttribute("name").toString().contains(feature)){
                                        found = true;
                                        String tmp_str = "<intent-filter>"+child_node_I.toString();
                                        String new_tmp = tmp_str.substring(0,tmp_str.length()-1).concat("/></intent-filter>");
                                        this.utility.WriteFile("xml_tag.xml",new_tmp);
                                        break;
                                    }
                                }
                            }
                        }
                        if(found){break;}
                    }
                    if(found){
                        this.utility.WriteFile("intent_class.txt",node.getAttribute("name").getValue().toString());

                        break;
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (XmlPullParserException e) {
            e.printStackTrace();
        }
    }
}
