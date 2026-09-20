import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import React, { useEffect, useMemo, useState } from 'react';

function DropdownMenu() {
  const [isOpen, setIsOpen] = useState(false);
  
  const [selectedOption, setSelectedOption] = useState('Select an option');

  // Toggle the open/closed state
  const handleToggle = () => setIsOpen(prev => !prev);

  // Handle option selection and close the menu
  const handleSelect = (option) => {
    setSelectedOption(option);
    setIsOpen(false);
  };

  return (
    <div className="relative inline-block text-left">
      {/* Trigger Button */}
      <button 
        onClick={handleToggle}
        className="inline-flex justify-between items-center w-48 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none"
      >
        <span>{selectedOption}</span>
      </button>

      {/* Dropdown List */}
      {isOpen && (
        <div className="absolute left-0 w-48 mt-2 bg-white border border-gray-200 rounded-md shadow-lg z-10">
          <div className="py-1">
            <button
              onClick={() => handleSelect('Option A')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Option A
            </button>
            <button
              onClick={() => handleSelect('Option B')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Option B
            </button>
            <button
              onClick={() => handleSelect('Option C')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Option C
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function SelectToTrack() {
  const [timeLeft, setTimeLeft] = useState(props.timeout || 6000);
  const [openDropdownId, setOpenDropdownId] = useState(null);
  const [editingText, setText] = useState("") 
  
  // Initialize checklist state dynamically based on props.items
  const [checkedItems, setCheckedItems] = useState(() => {
    const init = {};
    (props.items || []).forEach((item) => {
      init[item.id] = [item.label,item.description,item.defaultChecked || false, false];
    });
    return init;
  });

  // Countdown timer effect
  useEffect(() => {
    const interval = setInterval(() => {
      setTimeLeft((t) => (t > 0 ? t - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleEdit = (id) => {

    const finalText = editingText;
    console.log(finalText)
    setCheckedItems((prev) => ({
      ...prev,
      [finalText]: [finalText,prev[id][1],true,false],
    }));
    setCheckedItems((prev) => 
      Object.fromEntries(
        Object.entries(prev).filter(([key]) => key !== id)
    ));
    console.log(checkedItems)
    setText("")
    console.log(checkedItems)
  };

  // Add item handler
  const handleAddItem = () => {
    setCheckedItems((prev) => ({
      ...prev,
      ["New Topic"]: ["New Topic","Location",false,false],
    }));
    console.log(checkedItems)
  };


  // Handle individual checkbox state changes
  const handleEditToggle = (id, state) => {
    setCheckedItems((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],prev[id][2],state],
    }));
  };

  // Handle individual checkbox state changes
  const handleToggle = (id) => {
    setCheckedItems((prev) => ({
      ...prev,
      [id]: [prev[id][0],prev[id][1],!prev[id][2],false],
    }));
  };

  const handleSelectOption = (id, option) => {
    setCheckedItems((prev) => ({
      ...prev,
      [id]: [prev[id][0], option, true, false],
    }));
    setOpenDropdownId(null);
  };

  // Reset all checkboxes to their initial state
  const handleReset = () => {
    const init = {};
    (props.items || []).forEach((item) => {
      init[item.id] = [item.label,item.description,item.defaultChecked || false,false];
    });
    setCheckedItems(init);
    setOpenDropdownId(null);
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
          <div style={{display: 'grid', gridTemplateColumns: '20px 220px 340px', gap: '10px', 'column-gap': '4px', width: '100%'}} className="border-2 border-indigo-500 rounded-lg p-6 bg-card">
            {Object.entries(checkedItems).map(([id, item]) => {

              const isOpen = openDropdownId === id;
              const currentSelection = checkedItems[id][1]|| "Select option";

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
                {checkedItems[id][3] ? (<input 
                    type="text" 
                    value={editingText} 
                    onChange={(e) => setText(e.target.value)}
                    onBlur={() => handleEditToggle(id,false)} 
                    onFocus={(e) => setText(checkedItems[id][0])}
                    onKeyDown={(e) => e.key === 'Enter' && handleEdit(id)}
                    placeholder="Enter new name..."
                    autoFocus
                    className="border-2 px-2 rounded text-base focus:outline-none focus:ring-gray-50 bg-card text-muted-foreground"
                  />):
                   (<span className="text-base px-2 text-foreground font-semibold" onClick={() => handleEditToggle(id,true)} >{checkedItems[id][0]}</span>)}
              </div>


              {!(item.label=="Add More") ? (<div style={{display: 'flex', alignItems: 'center'}} className="col-span-1 py-1 border-indigo-100 relative">
                    <div className="w-full">
                      <button
                        type="button"
                        onClick={() => setOpenDropdownId(isOpen ? null : id)}
                        className="w-full flex justify-between items-center px-3 py-1.5 text-xs text-muted-foreground rounded-md shadow-sm hover:bg-gray-50 focus:outline-none"
                      >
                        <span className="truncate">{currentSelection}</span>
                        <svg className="w-4 h-4 ml-2 text-gray-400 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                      </button>

                      {isOpen && (
                        <div className="absolute left-0 right-0 mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-20">
                          <div className="py-1">
                            {(item.options || ["Character", "Location", "Faction", "Artifact", "Event"]).map((opt, optIdx) => (
                              <button
                                key={optIdx}
                                type="button"
                                onClick={() => handleSelectOption(id, opt)}
                                className="block w-full px-3 py-1.5 text-xs text-left text-gray-700 hover:bg-gray-100 truncate"
                              >
                                {opt}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>) : <div></div>}
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
          onClick={() => submitElement(checkedItems)}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  );
}