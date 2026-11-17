import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useMemo, useState } from 'react';


export default function NodeCard() {
  const [values, setValues] = useState(() => {
    const init = {};
    (props.fields || []).forEach((f) => {
      init[f.id] = f.value || '';
    });
    return init;
  });

  const [neighbor_idx, changePage] = useState(0)
  const [EditMode, setEditMode] = useState(false);
  const [node_layer, setLayer] = useState(props.type);
  const [node_text, setText] = useState(props.desc);
  const [catsums_text, setCatSumText] = useState(() => Object.entries(props.category_summaries || {}));
  const [EditCatMode, setEditCatMode] = useState(Object.entries(props.category_summaries).map((item) => false));
  const isExpanded = props.expanded
  const nextLayer = {
    "Agents & Collectives": "Places & Spaces",
    "Places & Spaces": "Objects & Artifacts",
    "Objects & Artifacts": "Concepts & Information",
    "Concepts & Information": "Events & Phenomena",
    "Events & Phenomena": "Agents & Collectives",
  };

  const handleClick = async (node) => {
    await callAction({
      name: "switch_node",
      payload: {node}
    });
  };

  const handleNodeChange = async (node_id) => {
    await callAction({
      name: "change_node",
      payload: {node_id}
    });
  };

  const handleShowEdges = async (node_id) => {
    await callAction({
      name: "show_edges",
      payload: {node_id}
    });
  };

  const handleLayerChange = async () => {
    setLayer(prevLayer => nextLayer[prevLayer] || "Agents & Collectives"); 
    await callAction({
      name: "update_node",
      payload: {"node_id":props.node_id,
                "key" : "type",
                "content":nextLayer[node_layer]}
    });   
  };

  const handleEditToggle = (state) => {
    setEditMode(state)
  };

  const setCatText = (target_index, text) => {
    setCatSumText((prev) =>
      prev.map((og_Cat, idx) => {
        if (idx !== target_index) return og_Cat;

      // Return a new copy of the target tuple with the updated element
      return og_Cat.map((item, colIndex) =>
        colIndex === 1 ? text : item
      );
     })
    );
  }

  const handleEditCatToggle = (state,target_index) => {
    console.log(state,target_index)
    console.log(EditCatMode)
    setEditCatMode((prev) => prev.map((isEditing, index) => (index === target_index ? state : isEditing)));
  };

  const handleCatEditComplete = async (key,index) => {
    await callAction({
      name: "update_node",
      payload: {"node_id":props.node_id,
                "key" : "cat_summary",
                "category":key,
                "content":catsums_text}
    });
  };

  const handlePageTurn = (dir) => {
    changePage(dir>0?neighbor_idx+5:neighbor_idx-5)
  };

  const handleEdit = async (node_id) => {
    const finalText = node_text;
    setText(finalText)
    await callAction({
      name: "update_node",
      payload: {"node_id":props.node_id,
                "key" : "summary",
                "content":node_text}
    });
  };


  const handleExit = async (save) => {
    console.log(node_layer)
    await callAction({
      name: "exit",
      payload: {save:save, new_desc:node_text, new_layer:node_layer}
    });
    await deleteElement();
  };

  const handleSave = async (save) => {
    console.log(node_layer)
    await callAction({
      name: "exit",
      payload: {save:save}
    });
  };

  const allValid = useMemo(() => {
    if (!props.fields) return true;
    return props.fields.every((f) => {
      if (!f.required) return true;
      const val = values[f.id];
      return val !== undefined && val !== '';
    });
  }, [props.fields, values]);

  const handleChange = (id, val) => {
    setValues((v) => ({ ...v, [id]: val }));
  };

  const handleReset = () => {
    const init = {};
    (props.fields || []).forEach((f) => {
      init[f.id] = f.value || '';
    });
    setValues(init);
  };

  return (
    <Card id="jira-ticket" className="mt-4 w-[340px] max-w-[340px]">
      <CardHeader className="space-y-1">
        <div style={{display: 'grid', gridTemplateColumns: '400px 240px', gap: '10px', 'column-gap': '4px', width: '100%'}}>
          <div>
            <CardTitle className="col-span-1">{props.name}</CardTitle>
          <span onClick={() => handleLayerChange()} className="cursor-pointer flex">
            <p className="text-sm font-medium text-muted-foreground">
              {node_layer}
            </p>
          </span>
          </div>
            <div style={{alignItems: 'right', textAlign:'right'}}>
              <Button id="ticket-cancel" variant="outline" onClick={() => handleShowEdges(props.node_id)}>
                Show Edges
              </Button>
            </div>
        </div>
        <div style={{padding:'4px'}}></div>
        <div style={{border: "2px solid #666666", 'border-radius':'8px', padding:'10px'}} className="col-span-2 flex-col space-y-2">
        <CardDescription className="col-span-2">
          {EditMode ? (<textarea 
                    rows={4}
                    type="text" 
                    value={node_text} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleEditToggle(false)} 
                    onFocus={(e) => setText(node_text)}
                    onKeyDown={(e) => {
                        // Optional: If you want 'Enter' to submit without adding a newline
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleEdit();
                            handleEditToggle(false);
                        }
                    }}
                    placeholder="Enter new name..."
                    autoFocus
                    className="border-2 rounded text-base focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground leading-normal"
                    style={{width:'100%', fontSize:'14px', wordBreak: 'break-word'}}
                  />):
                   (<span onClick={() => handleEditToggle(true)} >{node_text}</span>)}
        </CardDescription>
        </div>
      </CardHeader>

      <CardContent className="space-y-1">
        <div className="text-lg font-bold">Detailed Information</div>
        <div className="space-y-4">
          {catsums_text.map(([key,value],idx) =>(
          <div >
            {EditCatMode[idx] ? 
            (<textarea 
                    rows={4}
                    type="text" 
                    value={catsums_text[idx][1]} 
                    onChange={(e) => setCatText(idx, e.target.value)}
                    onBlur={() => handleEditCatToggle(false,idx)} 
                    onKeyDown={(e) => {
                        // Optional: If you want 'Enter' to submit without adding a newline
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleCatEditComplete(key,idx);
                            handleEditCatToggle(false,idx);
                        }
                    }}
                    placeholder="Enter new name..."
                    autoFocus
                    className="border-2 rounded text-base focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground leading-normal"
                    style={{width:'100%', fontSize:'14px', wordBreak: 'break-word'}}
                  />)
            : <p className="text-sm font-medium text-muted-foreground" onClick={() => handleEditCatToggle(true,idx)}> {value}</p>}
          </div>
          ))}
        </div>

        </CardContent>
        
      <CardFooter className="flex justify-end gap-2">
        <Button id="ticket-cancel" variant="outline" onClick={() => handleSave(true)}>
          Save
        </Button>
        <Button id="ticket-cancel" variant="outline" onClick={() => handleExit(false)}>
          Exit
        </Button>
      </CardFooter>
    </Card>
  );
}