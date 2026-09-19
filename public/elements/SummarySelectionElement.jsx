import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import React, { useEffect, useMemo, useState } from 'react';

export default function SummarySelectionElement() {
  const [timeLeft, setTimeLeft] = useState(props.timeout || 6000);
  const [openDropdownId, setOpenDropdownId] = useState(null);
  const [editingText, setText] = useState("") 
  
  // Initialize checklist state dynamically based on props.items
  const [checkedItems, setCheckedItems] = useState(() => {
    const init = {};
    (props.items || []).forEach((item) => {
      init[item.id] = [item.node,item.text,true, false,item.node_name];
    });
    return init;
  });

  const handleEdit = (id) => {

    const finalText = editingText;
    console.log(finalText)
    setCheckedItems((prev) => ({
      ...prev,
      [id]: [prev[id][0],finalText,true,false,prev[id][4]],
    }));
    console.log(checkedItems)
    setText("")
    console.log(checkedItems)
  };


  // Handle individual checkbox state changes
  const handleEditToggle = (id, state) => {
    setCheckedItems((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],prev[id][2],state,prev[id][4]],
    }));
  };

  // Handle individual checkbox state changes
  const handleToggle = (id) => {
    setCheckedItems((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],!prev[id][2],false,prev[id][4]],
    }));
  };

  // Reset all checkboxes to their initial state
  const handleReset = () => {
    const init = {};
    (props.items || []).forEach((item) => {
      init[item.id] = [item.node,item.text,true,false,item.node_name];
    });
    setCheckedItems(init);
    setOpenDropdownId(null);
    console.log(checkedItems)
  };



  return (
    <Card id="dynamic-checklist" className="mt-4 w-full max-w-2xl grid grid-cols-1 gap-4">
      <CardHeader className="space-y-2">
        <p className="text-sm font-medium text-muted-foreground">
          {"Please completetete all required items before submission."}
        </p>
        <CardTitle>{props.Title || "Action Checklist"}</CardTitle>
        <CardDescription>Complete the steps outlined below. {timeLeft}s left</CardDescription>
      </CardHeader>

      <CardContent className="w-full pt-2">
        {props.items && props.items.length > 0 ? (
          <div style={{display: 'grid', gridTemplateColumns: '20px 120px 300px', gap: '10px', 'column-gap': '4px', width: '100%'}} className="border-2 border-indigo-500 rounded-lg p-6 bg-card">
            {Object.entries(checkedItems).map(([id, item]) => {

              return(
             <React.Fragment key={id}>
              <div 
                key={id} 
                style={{display: 'flex', alignItems: 'center', textAlign:'center'}}
                className="col-span-1 space-x-3 py-2 border-indigo-500">
                <Checkbox
                  id={id}
                  checked={!!checkedItems[id][2]}
                  onCheckedChange={() => handleToggle(id)}
                  className="mt-1"/>
              </div>

              <div style={{display: 'flex', alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-1 ">
                {(<span className="text-base px-2 text-foreground " onClick={() => handleToggle(id)} >{checkedItems[id][4]}</span>)}
              </div>

              <div style={{display: 'flex', alignItems: 'center', textAlign:'center'}} className="col-span-1 space-x-3 py-1 ">
                {checkedItems[id][3] ? (<input 
                    type="text" 
                    value={editingText} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleEditToggle(id,false)} 
                    onFocus={(e) => setText(checkedItems[id][1])}
                    onKeyDown={(e) => e.key === 'Enter' && handleEdit(id)}
                    placeholder="Update Text"
                    autoFocus
                    className="w-full border-2 px-2 rounded text-base focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground"
                  />):
                   (<span className="text-base px-2 text-foreground " onClick={() => handleEditToggle(id,true)} >{checkedItems[id][1]}</span>)}
              </div>

              </React.Fragment>
            )})}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No checklist items available.</p>
        )}
      </CardContent>

      <CardFooter className="flex justify-end gap-2 border-t-2 border-indigo-500 pt-4">
        <Button id="checklist-reset" variant="ghost" onClick={handleReset}>
          Reset
        </Button>
        <Button id="checklist-cancel" variant="outline" onClick={() => cancelElement?.()}>
          Cancel
        </Button>
        <Button
          id="checklist-submit"
          onClick={() => submitElement(checkedItems)}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  );
}