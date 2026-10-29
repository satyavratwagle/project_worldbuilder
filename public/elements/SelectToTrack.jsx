import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import React, { useEffect, useMemo, useState } from 'react';


export default function SelectToTrack() {
  const allowedTypes = ["Unknown","Agents & Collectives","Places & Spaces","Objects & Artifacts","Concepts & Information","Events & Phenomena"]
  const [openDropdownId, setOpenDropdownId] = useState(null);
  const [editingText, setText] = useState("") 
  
  // Initialize checklist state dynamically based on props.items
  const [Topics, setTopics] = useState(() => {
    const widget = {};
    (props.items || []).forEach((topic) => {
      const current_item = structuredClone(topic);
      widget[topic.id] = topic
      widget[topic.id] = {...widget[topic.id],
                          "selected":true,
                          "editing_name":false,
                          "editing_tag":false,
                          "editing_alias":false,
                          "topic_idx":0}
    });
    console.log(widget)
    return widget;
  });

  // Handle individual checkbox state changes
  const handleEditToggle = (id, state) => {
    setTopics((prev) => {
      const topic = prev[id];
      return {
        ...prev,
        [id]: {...topic,editing_name: state
        }
      };
    });
  };
  
  const handleEdit = (id) => {
    const finalText = editingText;
    const newKey = Math.max(...Object.keys(Topics).map(Number));
    const uniqueAliases = [...new Set([finalText,...Topics[id].aliases])];
    console.log(uniqueAliases)
    const filteredAliases = uniqueAliases.filter((item) => item !== Topics[id].name);
    setTopics((prev) => {
      const topic = prev[id];
      return {
        ...prev,
        [id]: {...topic,name: finalText, aliases:filteredAliases
        }
      };
    });
    setText("")
  };

  const verifyTopicType = (id) =>{
    if(!allowedTypes.includes(Topics[id].type)){
      setTopics((prev) => {
      const topic = prev[id];
      return {
        ...prev,
        [id]: {...topic,type: allowedTypes[0], type_idx: 0
        }
      };
    });
    }
  }

  const handleTopicTypeChange = (id) =>{
    const curr_topic_idx = Topics[id].type_idx;
    setTopics((prev) => {
      const topic = prev[id];
      return {
        ...prev,
        [id]: {...topic,type: allowedTypes[curr_topic_idx+1], type_idx: curr_topic_idx+1
        }
      };
    });
  }

  // Add item handler
  const handleAddItem = () => {
    setCheckedItems((prev) => ({
      ...prev,
      [(Object.keys(checkedItems).length+1)]: ["New Topic",[],false,false],
    }));
    console.log(checkedItems)
  };


  // Handle individual checkbox state changes
  const handleToggle = (id) => {
    setTopics((prev) => {
      const topic = prev[id];
      return {
        ...prev,
        [id]: {...topic,selected: !topic.selected
        }
      };
    });
    console.log(Topics)
  };

  // Reset all checkboxes to their initial state
  const handleReset = () => {
    const widget = {};
    (props.items || []).forEach((topic) => {
      const current_item = structuredClone(topic);
      widget[topic.id] = topic
      widget[topic.id] = {...widget[topic.id],
                          "selected":true,
                          "editing_name":false,
                          "editing_tag":false,
                          "editing_alias":false}
    });
    setTopics(widget);
  };



  return (
    <Card id="dynamic-checklist" className="mt-4 w-full max-w-2xl grid grid-cols-1 gap-4">
      <CardHeader className="space-y-2">
        <p className="text-sm font-medium text-muted-foreground">
          {"Please completetete all required items before submission."}
        </p>
        <CardTitle>{"Please select the topics you want to track!"}</CardTitle>
        <CardDescription>Change this later.</CardDescription>
      </CardHeader>

      <CardContent className="w-full pt-2">
        {props.items && props.items.length > 0 ? (
          <div style={{display: 'grid', gridTemplateColumns: '20px 260px 300px', gap: '10px', 'column-gap': '4px', width: '100%'}} className="border-2 border-indigo-500 rounded-lg p-6 bg-card">
            {Object.entries(Topics).map(([id, topic]) => {
              const currentSelection = Topics[id].type||"Select Option";
              verifyTopicType(id);


              return(
                     <React.Fragment key={id}>
                      <div 
                        key={id} 
                        style={{display: 'flex', alignItems: 'center', textAlign:'center'}}
                        className="col-span-1 space-x-3 py-2 border-indigo-500">
                        <Checkbox
                          id={id}
                          checked={!!Topics[id].selected}
                          onCheckedChange={() => handleToggle(id)}
                          className="mt-1"/>
                      </div>


                      <div style={{alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-1 ">
                        <div className="w-full">
                        {Topics[id].editing_name ? (<input 
                            type="text" 
                            value={editingText} 
                            onChange={(e) => setText(e.target.value)}
                            onBlur={() => handleEditToggle(id,false)} 
                            onFocus={(e) => setText(Topics[id].name)}
                            onKeyDown={(e) => e.key === 'Enter' && handleEdit(id)}
                            placeholder="Enter new name..."
                            autoFocus
                            className="border-2 px-2 rounded text-base focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground"
                          />):
                           (<span className="text-base text-left px-2 text-foreground font-semibold" onClick={() => handleEditToggle(id,true)} >{Topics[id].name}</span>)}
                          </div>

                          <span className="text-xs text-muted-background" style={{opacity:0.7}}>{Topics[id].aliases.join('; ')}</span>
                      </div>

                      <div style={{alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-1 w-full">
                        <div className="w-full">
                              <span className="text-base text-muted-background" onClick={() => handleTopicTypeChange(id)}>{Topics[id].type} ({Topics[id].tag})</span>
                          </div>

                          <span className="text-xs text-muted-background" style={{opacity:0.7}}>{Topics[id].reasoning||"No reasoning available."}</span>
                      </div>
                      </React.Fragment>
                    )})}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No checklist items available.</p>
        )}
      </CardContent>

      <CardFooter className="flex justify-end gap-2 border-t-2 border-indigo-500 pt-4">
        <Button id="checklist-reset" variant="ghost" onClick={handleAddItem}>
          Add Topic
        </Button>
        <Button id="checklist-reset" variant="ghost" onClick={handleReset}>
          Reset
        </Button>
        <Button id="checklist-cancel" variant="outline" onClick={() => cancelElement?.()}>
          Cancel
        </Button>
        <Button
          id="checklist-submit"
          onClick={() => submitElement(Topics)}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  );
}