import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useMemo, useState } from 'react';


export default function EdgeCard() {
  const [edges, setEdges] = useState(() => props.edges || []);

  const [edge_idx, changePage] = useState(0)
  const [EditMode, setEditMode] = useState(props.edges.map((item) => false));
  const [node_layer, setLayer] = useState(props.type);
  const [edge_text, setEdgeText] = useState(props.edges.map((item) => item[4] || "None"));
  const [edge_layer, setEdgeLayer] = useState(props.edges.map((item) => item[5] || "Structural & Compositional"));
  const [temp_text, setTempText] = useState("");
  const isExpanded = props.expanded
  const nextLayer = {
    "Structural & Compositional":"Spatial & Topological",
    "Spatial & Topological":"Temporal & Causal",
    "Temporal & Causal":"Social & Organizational",
    "Social & Organizational":"Functional & Operational",
    "Functional & Operational": "Structural & Compositional",
  };

  const handleEdgeLayerChange = async (target_index) => {
    setEdgeLayer((prevLayer) =>prevLayer.map((og_layer, index) => (index === target_index ? nextLayer[og_layer] : og_layer)));
    await callAction({
      name: "update_edge",
      payload: {"node_id":props.node_id,
                "index":target_index,
                "content_key" : "type",
                "content":nextLayer[edge_layer[target_index]]}
    });
  };

  const handleRemoveEdge = async (target_index) => {
    await callAction({
      name: "update_edge",
      payload: {"node_id":props.node_id,
                "index":target_index,
                "content_key" : "delete",
                "content":"None"}
    });
  };

  const handleClick = async (node) => {
    await callAction({
      name: "switch_node",
      payload: {node}
    });
  };

  const handleShowNode = async (node_id) => {
    await callAction({
      name: "show_node",
      payload: {node_id}
    });
  };

  const handleNodeChange = async (node_id) => {
    console.log(props)
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
    setLayer(prevLayer => nextLayer[prevLayer] || "Agents & Collectives");    
  };

  const handleEditToggle = (target_index,state) => {
    console.log(target_index,state)
    console.log(EditMode)
    setEditMode((prev) => prev.map((isEditing, index) => (index === target_index ? state : isEditing)));
  };

  const handlePageTurn = (dir) => {
    changePage(dir>0?edge_idx+5:edge_idx-5)
    console.log(edge_idx)
  };

  const setElementText = (target_index, text) => {
    console.log(text)
    setEdgeText((prev) =>
      prev.map((og_text, idx) => idx === target_index ? text : og_text)
    );
    console.log(edge_text[target_index])
  }

  const handleEdit = async (target_index) => {
    await callAction({
      name: "update_edge",
      payload: {"node_id":props.node_id,
                "index":target_index,
                "content_key" : "text",
                "content":edge_text[target_index]}
    });
  };

  const handleExit = async (save) => {
    await callAction({
      name: "exit",
      payload: {save:save}
    });
    await deleteElement();
  };

  const handleSave = async (save) => {
    await callAction({
      name: "exit",
      payload: {save:save}
    });
  };

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
              {props.type}
            </p>
          </span>
          </div>
            <div style={{alignItems: 'right', textAlign:'right'}}>
              <Button id="ticket-cancel" variant="outline" onClick={() => handleShowNode(props.node_id)}>
                Show Node
              </Button>
            </div>
        </div>
        <div style={{padding:'4px'}}></div>
        <div className="col-span-2 flex-col space-y-2">
        <CardDescription className="col-span-2">

          {Object.entries(props.edges).slice(edge_idx,edge_idx+5).map(([rawIndex]) => {
            const index = Number(rawIndex);
            return (
            <div style={{display: 'grid', gridTemplateColumns: '200px 300px 80px', gap: '10px', 'column-gap': '4px', width: '100%'}} className="border-2 border-indigo-500 rounded-lg p-6 bg-card">
              <React.Fragment key={index}>
                <div>
                  <div style={{alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-1 ">
                    <Button variant="ghost" className="border-2 border-indigo-500 rounded-lg" onClick={() => handleShowNode(props.edges[index][3])}>{props.edges[index][3]}</Button>
                  </div>
                  <div style={{alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-1 ">
                    <span className="text-xs text-muted-background" onClick={() => handleEdgeLayerChange(index)}>{edge_layer[index]}</span>
                  </div>
                </div>
                <div style={{alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-1 ">
                  {EditMode[index] ? (<textarea 
                    rows={2}
                    type="text" 
                    value={edge_text[index]} 
                    onChange={(e) => setElementText(index,e.target.value)}
                    onBlur={() => handleEditToggle(index,false)} 
                    onKeyDown={(e) => {
                        // Optional: If you want 'Enter' to submit without adding a newline
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleEdit(index);
                            handleEditToggle(index,false);
                        }
                    }}
                    placeholder="Enter new name..."
                    autoFocus
                    className="border-2 rounded text-base focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground leading-normal"
                    style={{width:'100%', fontSize:'14px', wordBreak: 'break-word'}}
                  />):
                   (<span onClick={() => handleEditToggle(index,true)} >{edge_text[index]}</span>)}

                </div>
                <div style={{alignItems: 'right', textAlign:'right'}}>
                  <Button id="ticket-cancel" variant="destructive" onClick={() => handleRemoveEdge(index)}>
                    Delete
                  </Button>
                </div>

              </React.Fragment>
            </div>
            )})}
          <div style={{padding:'6px'}}></div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Button variant="ghost" disabled={edge_idx==0} onClick={() => handlePageTurn(-1)}>
                Prev
              </Button>
              <Button variant="ghost" disabled={edge_idx+5>=props.edges.length} onClick={() => handlePageTurn(+1)}>
                Next
              </Button>
            </div>
            <div></div>
          </div>
        </CardDescription>
        </div>
      </CardHeader>

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