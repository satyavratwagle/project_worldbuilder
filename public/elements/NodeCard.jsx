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
  const [node_layer, setLayer] = useState(props.layer);
  const [node_text, setText] = useState(props.desc);
  const isExpanded = props.expanded
  const nextLayer = {
    'Lore': 'Topic',
    'Topic': 'Plot',
    'Plot': 'Lore'
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

  const handleStackChange = async (node_id,idx,rem) => {
    await callAction({
      name: "stack",
      payload: {node_id,idx,rem}
    });
  };

  const handleLayerChange = () => {
    setLayer(prevLayer => nextLayer[prevLayer] || 'Lore');    
  };

  const handleEditToggle = (state) => {
    setEditMode(state)
  };

  const handlePageTurn = (dir) => {
    changePage(dir>0?neighbor_idx+5:neighbor_idx-5)
  };

  const handleEdit = () => {
    const finalText = node_text;
    console.log(finalText)
    setText(finalText)
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
      payload: {save:save, new_desc:node_text, new_layer:node_layer}
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
        <div>
          <span onClick={() => handleLayerChange()} className="cursor-pointer flex">
            <p className="text-sm font-medium text-muted-foreground">
              {node_layer}
            </p>
          </span>
        </div>
        <CardTitle className="col-span-1">{props.name}</CardTitle>
        <div style={{padding:'8px'}}></div>
        <div style={{border: "2px solid #666666", 'border-radius':'8px', padding:'10px'}} className="col-span-2 flex-col space-y-2">
        <CardDescription className="col-span-2">
          {EditMode ? (<textarea 
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
                    className="border-2 rounded text-base focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground"
                    style={{width:'100%', fontSize:'14px', wordBreak: 'break-word'}}
                  />):
                   (<span onClick={() => handleEditToggle(true)} >{node_text}</span>)}
        </CardDescription>
        </div>
      </CardHeader>

      <CardContent className="space-y-1">
        <div className="flex-col space-y-2">
          <div className="text-lg font-bold">Related Nodes</div>
          {Object.entries(props.neighbors).slice(neighbor_idx,neighbor_idx+5).map(([idx]) =>(
              <div>
                <span onClick={() => handleNodeChange(props.neighbors[idx][2])} className="cursor-pointer flex">
                  <p className="text-sm font-medium text-muted-foreground"> • (<b>{props.neighbors[idx][0]}</b>) {props.neighbors[idx][1]}</p>
                </span>
              </div>
            ))
          }
        </div>
        <div style={{padding:'6px'}}></div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Button variant="ghost" disabled={neighbor_idx==0} onClick={() => handlePageTurn(-1)}>
              Prev
            </Button>
            <Button variant="ghost" disabled={neighbor_idx+5>=props.neighbors.length} onClick={() => handlePageTurn(+1)}>
              Next
            </Button>
          </div>
          <div></div>
        </div>

        {props.stack.length>0 && (
        <div>
          <div className="text-lg font-bold">Stack</div>
          {Object.entries(props.stack).map(([idx]) =>(
          <div>
            <p className="text-sm font-medium text-muted-foreground"> {parseInt(idx,10)+1}. (<b>{props.stack[idx][0]}</b>) {props.stack[idx][1]}</p>
          </div>
          ))}
        </div>
        )}
        </CardContent>
        
      <CardFooter className="flex justify-end gap-2">
        <Button id="ticket-cancel" variant="outline" onClick={() => handleStackChange(props.node_id,props.stack.length,false)}>
          Stack
        </Button>
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