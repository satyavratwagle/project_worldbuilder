import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import React, { useEffect, useMemo, useState } from 'react';

export default function SelectToTrack() {
  const [timeLeft, setTimeLeft] = useState(props.timeout || 30);
  
  // Initialize checklist state dynamically based on props.items
  const [checkedItems, setCheckedItems] = useState(() => {
    const init = {};
    (props.items || []).forEach((item) => {
      init[item.id] = item.defaultChecked || false;
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

  // Handle individual checkbox state changes
  const handleToggle = (id) => {
    setCheckedItems((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  // Reset all checkboxes to their initial state
  const handleReset = () => {
    const init = {};
    (props.items || []).forEach((item) => {
      init[item.id] = item.defaultChecked || false;
    });
    setCheckedItems(init);
  };

  // Ensure all required checklist items are checked before enabling submission
  const allRequiredChecked = useMemo(() => {
    if (!props.items) return true;
    return props.items.every((item) => {
      if (!item.required) return true;
      return checkedItems[item.id];
    });
  }, [props.items, checkedItems]);

  return (
    <Card id="dynamic-checklist" className="mt-4 w-full max-w-2xl grid grid-cols-1 gap-4">
      <CardHeader className="space-y-2">
        <p className="text-sm font-medium text-muted-foreground">
          {props.topText || "Please complete all required items before submission."}
        </p>
        <CardTitle>{props.Title || "Action Checklist"}</CardTitle>
        <CardDescription>Complete the steps outlined below. {timeLeft}s left</CardDescription>
      </CardHeader>

      <CardContent className="w-full pt-2">
        {props.items && props.items.length > 0 ? (
          <div className="flex flex-col gap-1 w-full border-2 border-indigo-500 rounded-lg p-6 bg-card">
            {props.items.map((item) => (
              <div 
                key={item.id} 
                className="flex items-start space-x-3 py-1 border-b border-indigo-100 last:border-b-0"
              >
                <Checkbox
                  id={item.id}
                  checked={!!checkedItems[item.id]}
                  onCheckedChange={() => handleToggle(item.id)}
                  className="mt-1"
                />
                <div className="grid grid-cols-2 w-full leading-normal">
                  <Label
                    htmlFor={item.id}
                    className="text-sm font-semibold leading-normal justify-end px-3 peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                  >
                    {item.label}
                    {item.required && <span className="text-red-500 ml-1">*</span>}
                  </Label>
                  {item.description && (
                    <p className="text-xs text-muted-foreground justify-start">
                      {item.description}
                    </p>
                  )}
                </div>
              </div>
            ))}
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
          disabled={!allRequiredChecked}
          onClick={() => submitElement(checkedItems)}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  );
}