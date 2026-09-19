import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useMemo, useState } from 'react';


function SentenceLink({ sentence, targetWords, onWordClick }) {
  if (!sentence) return null;

  const sortedWords = [...targetWords].sort((a, b) => b.length - a.length);
  const escapedWords = sortedWords.map(w => w.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));
  const regex = new RegExp(`\\b(${escapedWords.join('|')})\\b`, 'g');
  const parts = sentence.split(regex);
  console.log(props.wiki)

  return (
    <p>
      {parts.map((part, index) => {
        if (targetWords.includes(part)) {
          return (
            <span
              key={index}
              onClick={() => onWordClick(part)}
              className="cursor-pointer text-blue-600 underline hover:text-blue-800 font-medium"
            >
              <u>{part}</u>
            </span>
          );
        }
        return part;
      })}
    </p>
  );
}

export default function GraphInfo() {
  const [values, setValues] = useState(() => {
    const init = {};
    (props.fields || []).forEach((f) => {
      init[f.id] = f.value || '';
    });
    return init;
  });

  const handleClick = async (node) => {
    await callAction({
      name: "switch_node",
      payload: {node}
    });
  };


  const handleNodeChange = async (node,summary) => {
    await callAction({
      name: "update_node_summary",
      payload: {node,summary}
    });
  };

  const handleDelete = async (head,tail,key,text) => {
    await callAction({
      name: "delete_edge",
      payload: {head,tail,key,text}
    });
  };

  const handleDeleteNode = async (node) => {
    await callAction({
      name: "delete_node",
      payload: {node}
    });
  };

  const handleEdit = async (head,tail,key,text) => {
    await callAction({
      name: "edit_edge",
      payload: {head,tail,key,text}
    });
  };

  const handleExit = async (save) => {
    await callAction({
      name: "exit",
      payload: {save}
    });
  };

  const handleCallback = (clickedWord) => {
    console.log(`Action triggered from word: ${clickedWord}`);
    // Put your custom logic, API call, or state update here
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
      <CardHeader style={{display: 'grid', gridTemplateColumns: '80% 20%', width: '100%'}} className="space-y-1">
        <CardTitle className="col-span-1">{props.title || "Edge Information"}</CardTitle>
        <span style={{display: 'flex', alignItems: 'center', border: "1px solid #666666", 'border-radius':'8px', padding:'4px'}} onClick={() => handleDeleteNode(props.title)} className="col-span-1 cursor-pointer underline text-sm font-tiny text-muted-background justify-center">{"Delete Node"}</span>
        <CardDescription className="col-span-2"><span onClick={() => handleNodeChange(props.title,props.subtitle)} className="cursor-pointer">{props.subtitle || ""}</span></CardDescription>
      </CardHeader>

      <CardContent style={{display: 'grid', gridTemplateColumns: '440px 20px', gap: '10px', 'column-gap': '4px', width: '100%'}} className="space-y-1">
        <div style={{border: "2px solid #666666", 'border-radius':'8px', padding:'10px'}} className="col-span-2 flex-col space-y-2">
          {Object.entries(props.wiki).map(([key, value]) =>(
              <div>
                  <div className="text-lg font-bold">{key.charAt(0).toUpperCase() + key.slice(1)}</div>
                  <p className="text-sm font-medium text-muted-foreground">{value}</p>
              </div>
            ))
          }
        </div>
        <hr style={{border: "10px"}} className="col-span-2"/>
        {props.edges.map(([head, tail, key,text, endpoints], index) => (
          <React.Fragment key={index}>
            <div>
              <div style={{padding: "8px", 'border-radius': '16px', display: 'flex', alignItems: 'center'}} className="col-span-1 text-left text-sm text-muted-foreground">
                <SentenceLink 
                  sentence={text} 
                  targetWords={[endpoints[0],endpoints[1]].filter(word=>!(word==props.title))} 
                  onWordClick={(word) => handleClick(word)} 
                /></div>
          </div>
          <div style={{display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{display: 'flex', alignItems: 'center', textAlign:'center'}} className="col-span-1"><span onClick={() => handleEdit(head,tail,key,text)} className="cursor-pointer underline">✎</span></div>
            <hr style={{border: "1px solid #666666"}}/>
            <div style={{display: 'flex', marginTop: 'auto'}} className="col-span-1"><span onClick={() => handleDelete(head,tail,key,text)} className="cursor-pointer flex items-center justify-center">✖</span></div>
          </div>
          <hr style={{border: "1px solid #666666"}} className="col-span-2" />
          </React.Fragment>
        ))}
      </CardContent>
      <CardFooter className="flex justify-end gap-2">
        <Button id="ticket-cancel" variant="outline" onClick={() => handleExit(false)}>
          Exit
        </Button>
        <Button id="ticket-cancel" variant="outline" onClick={() => handleExit(true)}>
          Save & Exit
        </Button>
      </CardFooter>
    </Card>
  );
}