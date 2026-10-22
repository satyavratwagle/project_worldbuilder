import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useMemo, useState } from 'react';

export default function TripletListWidget() {
  const [editingText, setText] = useState("")

  const [Triplets, setTriplets] = useState(() => {
    const widget = {};

    // props = {triplets:[{id:head,relation,tail,base_fact}]}

    // Each triplet should have a [0] : head (str), [1] : relation (str), [2] : tail (str), [3] : base_fact(str), 
    // [4] : select (bool), [5] : editing_fact (bool), [6] : editing_relation (bool)
    (props.triplets || []).forEach((triplet) => {
      widget[triplet.id] = [triplet.head,triplet.relation,triplet.tail,triplet.base_fact,true,false,false];
    });
    console.log(props)
    return widget;
  });

  // Handle Swap
  const handleSwap = (id) => {
    setTriplets((prev) => ({
      ...prev,
      [id]: [prev[id][2],prev[id][1],prev[id][0],prev[id][3],prev[id][4],prev[id][5],prev[id][6]],
    }));
  };

  // Handle Relation Edit Toggle
  const handleRelationEditToggle = (id, state) => {
    setTriplets((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],prev[id][2],prev[id][3],prev[id][4],prev[id][5],state],
    }));
  };

  // Handle Relation Edit
  const handleRelationEdit = (id) => {
    const finalText = editingText;
    setTriplets((prev) => ({
      ...prev,
      [id]: [prev[id][0],finalText,prev[id][2],prev[id][3],prev[id][4],prev[id][5],false],
    }));
    setText("")
  };

  // Handle Base Fact Edit Toggle
  const handleFactEditToggle = (id, state) => {
    setTriplets((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],prev[id][2],prev[id][3],prev[id][4],state,prev[id][6]],
    }));
  };

  // Handle Base Fact Edit
  const handleFactEdit = (id) => {

    const finalText = editingText;
    setTriplets((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],prev[id][2],finalText,prev[id][4],false,prev[id][6]],
    }));
    setText("")
  };

  // Handle Select
  const handleSelect = (id) => {
    setTriplets((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],prev[id][2],prev[id][3],!prev[id][4],prev[id][5],prev[id][6]],
    }));
  };

  const handleChange = (id, val) => {
    setTriplets((v) => ({ ...v, [id]: val }));
  };

  const handleReset = () => {
    const widget = {};
    (props.triplets || []).forEach((triplet) => {
      widget[triplet.id] = [triplet.head,triplet.relation,triplet.tail,triplet.base_fact,true,false,false];
    });
    setTriplets(widget);
  };


  return (
    <Card id="jira-ticket" className="mt-4 w-full max-w-2xl grid grid-cols-1 gap-4">
      <CardHeader className="col-span-2 space-y-1">
        <p className="text-sm font-medium text-muted-foreground">
          {"Please Check Triplet Information!"}
        </p>
      </CardHeader>

      <CardContent className="w-full pt-2">
        {props.triplets && props.triplets.length > 0 ? (
          <div style={{display: 'grid', gridTemplateColumns: '20px 560px', gap: '10px', 'column-gap': '4px', width: '100%'}} className="border-2 border-indigo-500 rounded-lg p-6 bg-card">
            {Object.entries(Triplets).map(([id, item]) => {
              return(
             <React.Fragment key={id}>
              <div key={id} style={{display: 'flex', alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-2 border-indigo-500">
                <Checkbox id={id} checked={!!Triplets[id][4]} onCheckedChange={() => handleSelect(id)} className="mt-1 border-2"/>
              </div>

              <div className={`col-span-1 flex flex-col items-start space-y-1 py-2 px-2 ${
                              Triplets[id][4] 
                                ? 'border-2 rounded-lg' 
                                : ''
                            }`}>

                <div className="flex items-center space-x-3 w-full">
                <span className="text-base text-left px-2 text-foreground font-semibold" onClick={() => handleSwap(id)}> {Triplets[id][0]}</span>

                {/* Arrow Container with Absolute Label Positioning */}
                <div className="flex-1 relative flex items-center">
                  
                  {/* Relation Label: Floats absolutely above the line, centered */}
                  {Triplets[id][6] ? (<input 
                    type="text" 
                    value={editingText} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleRelationEditToggle(id,false)} 
                    onFocus={(e) => setText(Triplets[id][1])}
                    onKeyDown={(e) => e.key === 'Enter' && handleRelationEdit(id)}
                    placeholder="Update the description!"
                    autoFocus
                    className="border-2 px-1 rounded text-s text-center w-full focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground absolute left-1/2 -translate-x-1/2 -top-3.5"
                  />):
                    (<span className="absolute left-1/2 -translate-x-1/2 -top-3.5 text-s text-foreground bg-card px-1 whitespace-nowrap" onClick={() => handleRelationEditToggle(id,true)} >{Triplets[id][1]}</span>)}
                  
                  {/* The Line and Arrowhead (Locked to the vertical center) */}
                  <div className="w-full flex items-center">
                    <div className="flex-1 border-t border-muted-foreground/60" />
                    <span className="text-muted-foreground text-xs leading-none ml-0.5">▶</span>
                  </div>

                </div>

                <span className="text-base text-left px-2 text-foreground font-semibold" onClick={() => handleSwap(id)}> {Triplets[id][2]}</span>
              </div>
                
                

                {Triplets[id][5] ? (<input 
                    type="text" 
                    value={editingText} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleFactEditToggle(id,false)} 
                    onFocus={(e) => setText(Triplets[id][3])}
                    onKeyDown={(e) => e.key === 'Enter' && handleFactEdit(id)}
                    placeholder="Update the description!"
                    autoFocus
                    className="border-2 px-2 rounded text-sm text-center w-full focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground"
                  />):
                    (<span className="w-full text-center text-sm text-muted-foreground px-2" onClick={() => handleFactEditToggle(id,true)} >{Triplets[id][3]}</span>)}
              </div>

              </React.Fragment>
            )})}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No triplets available.</p>
        )}
      </CardContent>

      <CardFooter className="col-span-2 flex justify-end gap-2">
        <Button id="ticket-reset" variant="ghost" onClick={handleReset}>
          Reset
        </Button>
        <Button id="ticket-cancel" variant="outline" onClick={() => props.onCancel?.()}>
          Cancel
        </Button>
        <Button
          id="ticket-submit"
          onClick={() => submitElement(Triplets)}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  );
}