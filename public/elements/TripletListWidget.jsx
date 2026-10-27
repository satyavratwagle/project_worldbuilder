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
    // [4] : select (bool), [5] : editing_fact (bool), [6] : editing_relation (bool), [7] : Head selection (list(bool)), [8] : Tail selection (list(bool))
    // [9] : temporal_metadata

    (props.triplets || []).forEach((triplet) => {
      //widget[triplet.id] = [triplet.head,triplet.relation,triplet.tail,triplet.base_fact,true,false,false,Object.fromEntries(triplet.head.map((item, index) => [index, true])),Object.fromEntries(triplet.tail.map((item, index) => [index, true])), triplet.temporal_metadata];
      const current_triplet = structuredClone(triplet);
      widget[triplet.id] = current_triplet
      widget[triplet.id] = {...widget[triplet.id],
                          "selected":true,
                          "editing_fact":false,
                          "editing_relation":false,
                          "head_selection":Object.fromEntries(triplet.head.map((item, index) => [index, true])),
                          "tail_selection":Object.fromEntries(triplet.tail.map((item, index) => [index, true]))}
    });
    console.log(widget)
    return widget;
  });

  // Handle Head Selection
  const handleHeadSelect = (id,head_id) => {
    setTriplets((prev) => {
      const triplet = prev[id];
      return {
        ...prev,
        [id]: {
          ...triplet,
          head_selection:{
            ...triplet.head_selection,
            [head_id]: !triplet.head_selection[head_id]
          }
        }
      };
    });
  };

  const handleTailSelect = (id,tail_id) => {
    setTriplets((prev) => {
      const triplet = prev[id];
      return {
        ...prev,
        [id]: {
          ...triplet,
          tail_selection:{
            ...triplet.tail_selection,
            [tail_id]: !triplet.tail_selection[tail_id]
          }
        }
      };
    });
  };

  // Handle Relation Edit Toggle
  const handleRelationEditToggle = (id, state) => {
    setTriplets((prev) => {
      const triplet = prev[id];
      return {
        ...prev,
        [id]: {...triplet,editing_relation: state
        }
      };
    });
  };

  // Handle Relation Edit
  const handleRelationEdit = (id) => {
    const finalText = editingText;
    setTriplets((prev) => {
      const triplet = prev[id];
      return {
        ...prev,
        [id]: {...triplet,relation: finalText
        }
      };
    });
    setText("")
  };

  // Handle Base Fact Edit Toggle
  const handleFactEditToggle = (id, state) => {
    setTriplets((prev) => {
      const triplet = prev[id];
      return {
        ...prev,
        [id]: {...triplet,editing_fact: state
        }
      };
    });
  };

  // Handle Base Fact Edit
  const handleFactEdit = (id) => {

    const finalText = editingText;
    setTriplets((prev) => {
      const triplet = prev[id];
      return {
        ...prev,
        [id]: {...triplet,base_fact: finalText
        }
      };
    });
    setText("")
  };

  // Handle Select
  const handleSelect = (id) => {
    setTriplets((prev) => {
      const triplet = prev[id];
      return {
        ...prev,
        [id]: {...triplet,selected: !triplet.selected
        }
      };
    });
  };

  const handleReset = () => {
    const widget = {};
    (props.triplets || []).forEach((triplet) => {
      //widget[triplet.id] = [triplet.head,triplet.relation,triplet.tail,triplet.base_fact,true,false,false,Object.fromEntries(triplet.head.map((item, index) => [index, true])),Object.fromEntries(triplet.tail.map((item, index) => [index, true])), triplet.temporal_metadata];
      const current_triplet = structuredClone(triplet);
      widget[triplet.id] = current_triplet
      widget[triplet.id] = {...widget[triplet.id],
                          "selected":true,
                          "editing_fact":false,
                          "editing_relation":false,
                          "head_selection":Object.fromEntries(triplet.head.map((item, index) => [index, true])),
                          "tail_selection":Object.fromEntries(triplet.tail.map((item, index) => [index, true]))}
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
                <Checkbox id={id} checked={!!Triplets[id].selected} onCheckedChange={() => handleSelect(id)} className="mt-1 border-2"/>
              </div>
              <div className={`col-span-1 flex flex-col items-start space-y-1 py-2 px-2 ${Triplets[id].selected ? 'border-2 rounded-lg' : ''}`}>

                <div style={{display: 'grid', gridTemplateColumns: '100px 330px 100px', gap: '10px', 'column-gap': '4px', width: '100%'}} className="flex items-center space-x-3 w-full">
                  <div className="flex flex-col space-y-1 w-full">
                    {Object.entries(Triplets[id].head).map(([head_id, head]) => {
                      return(
                      <React.Fragment key={`head-${head_id}`}>
                      <span className={`text-left px-2`} style={{textDecoration: Triplets[id].head_selection[head_id] ? 'none' : 'line-through',fontWeight: Triplets[id].head_selection[head_id] ? '600' : 'normal', opacity: Triplets[id].head_selection[head_id] ? 1 : 0.4}} onClick={() => handleHeadSelect(id,head_id)}> {head}</span>
                      </React.Fragment>
                    )})}
                  </div>


                {/* Arrow Container with Absolute Label Positioning */}
                <div className="flex-1 relative flex items-center">
                  
                  {/* Relation Label: Floats absolutely above the line, centered */}
                  {Triplets[id].editing_relation ? (<input 
                    type="text" 
                    value={editingText} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleRelationEditToggle(id,false)} 
                    onFocus={(e) => setText(Triplets[id].relation)}
                    onKeyDown={(e) => e.key === 'Enter' && handleRelationEdit(id)}
                    placeholder="Update the description!"
                    autoFocus
                    className="border-2 px-1 rounded text-s text-center w-full focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground absolute left-1/2 -translate-x-1/2 -top-3.5"
                  />):
                    (<span className="absolute left-1/2 -translate-x-1/2 -top-3.5 text-s text-foreground bg-card px-1 whitespace-nowrap" onClick={() => handleRelationEditToggle(id,true)} >{Triplets[id].relation}</span>)}
                  
                  {/* The Line and Arrowhead (Locked to the vertical center) */}
                  <div className="w-full flex items-center">
                    <div className="flex-1 border-t border-muted-foreground/60" />
                    <span className="text-muted-foreground text-xs leading-none ml-0.5">▶</span>
                  </div>

                </div>

          
                <div className="flex flex-col space-y-1 w-full">
                    {Object.entries(Triplets[id].tail).map(([tail_id, tail]) => {
                      return(
                      <React.Fragment key={`tail-${tail_id}`}>
                      <span className={`text-left px-2`} style={{textDecoration: Triplets[id].tail_selection[tail_id] ? 'none' : 'line-through', fontWeight: Triplets[id].tail_selection[tail_id] ? '600' : 'normal', opacity: Triplets[id].tail_selection[tail_id] ? 1 : 0.6}} onClick={() => handleTailSelect(id,tail_id)}> {tail}</span>
                      </React.Fragment>
                    )})}
                </div>

              </div>

                {Triplets[id].editing_fact ? (<input 
                    type="text" 
                    value={editingText} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleFactEditToggle(id,false)} 
                    onFocus={(e) => setText(Triplets[id].base_fact)}
                    onKeyDown={(e) => e.key === 'Enter' && handleFactEdit(id)}
                    placeholder="Update the description!"
                    autoFocus
                    className="border-2 px-2 rounded text-sm text-center w-full focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground"
                  />):
                    (<span className="w-full text-center text-sm text-muted-foreground px-2" onClick={() => handleFactEditToggle(id,true)} >{Triplets[id].base_fact}</span>)}

              {Triplets[id].editing_metadata ? (<input 
                    type="text" 
                    value={editingText} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleFactEditToggle(id,false)} 
                    onFocus={(e) => setText(Triplets[id].temporal_metadata)}
                    onKeyDown={(e) => e.key === 'Enter' && handleFactEdit(id)}
                    placeholder="Update the description!"
                    autoFocus
                    className="border-2 px-2 rounded text-sm text-center w-full focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground"
                  />):
                    (<span className="w-full text-center text-sm text-muted-foreground px-2" onClick={() => handleFactEditToggle(id,true)} >{Triplets[id].temporal_metadata}</span>)}
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