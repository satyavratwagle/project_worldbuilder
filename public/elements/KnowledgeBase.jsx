import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import React, { useEffect, useMemo, useState } from 'react';

export default function KnowledgeBase() {
  const [timeLeft, setTimeLeft] = useState(props.timeout || 30);
  const [values, setValues] = useState(() => {
    const init = {};
    (props.fields || []).forEach((f) => {
      init[f.id] = f.value || '';
    });
    return init;
  });

  // Track descriptions in state so they can be edited globally
  const [descriptions, setDescriptions] = useState(() => {
    const init = {};
    (props.fields || []).forEach((f) => {
      init[f.id] = [...(f.description || [])];
    });
    return init;
  });

  // Single global flag to control edit mode for ALL descriptions
  const [isEditingAllDesc, setIsEditingAllDesc] = useState(props.initEdit||false);

  const allValid = useMemo(() => {
    if (!props.fields) return true;
    return props.fields.every((f) => {
      if (!f.required) return true;
      const val = values[f.id];
      return val !== undefined && val !== '';
    });
  }, [props.fields, values]);

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeLeft((t) => (t > 0 ? t - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleChange = (id, val) => {
    setValues((v) => ({ ...v, [id]: val }));
  };

  const submitProcess = () => {
    (props.fields || []).forEach((f) => {
      const val = values[f.id];
      descriptions[f.id].push(val);
      return true; // .every() expects a boolean return value to continue iteration
    });
    return submitElement(descriptions);
  };

  const handleReset = () => {
    const initValues = {};
    const initDescriptions = {};
    (props.fields || []).forEach((f) => {
      initValues[f.id] = f.value || '';
      initDescriptions[f.id] = [...(f.description || [])];
    });
    setValues(initValues);
    setDescriptions(initDescriptions);
  };

  const handleDescriptionChange = (id, index, val) => {
    setDescriptions((prev) => {
      const list = [...(prev[id] || [])];
      list[index] = val;
      return { ...prev, [id]: list };
    });
  };

  const toggleAllDescriptionsEdit = () => {
    setIsEditingAllDesc((prev) => !prev);
  };

  const renderField = (field) => {
    const value = values[field.id];
    return <Input id={field.id} value={value} onChange={(e) => handleChange(field.id, e.target.value)} className="w-full block" />;
    };

  const defaultTab = useMemo(() => {
    if (!props.fields || props.fields.length === 0) return "";
    if (props.initialTab && props.fields.some((f) => f.id === props.initialTab)) {
      return props.initialTab;
    }
    return props.fields[1].id;
  }, [props.fields, props.initialTab]);

  return (
    <Card id="jira-ticket" className="w-full max-w-7xl grid grid-cols-1">
      <CardHeader className="col-span-1 space-y-2">
        <p className="text-sm font-medium text-muted-foreground">
          {" "}
        </p>
        <CardTitle>{props.Title}</CardTitle>
        <CardDescription>{props.topText}</CardDescription>
      </CardHeader>

      <CardContent className="col-span-1 grid grid-cols-4 block pt-2">
        {props.fields && props.fields.length > 0 ? (
          <Tabs defaultValue={defaultTab} className="grid grid-cols-4 rounded-lg">
              {/* Left Sidebar TabsList */}
              <TabsList className="flex-col h-full justify-start p-0 bg-dark rounded-lg border-indigo-500">
                {props.fields.map((field) => (
                  <TabsTrigger 
                    key={field.id} 
                    value={field.id} 
                    className="justify-start px-4 py-3 text-left w-full rounded-lg data-[state=inactive]:bg-dark"
                  >
                    {field.label}
                    {field.required && <span className="text-red-500 ml-1">*</span>}
                  </TabsTrigger>
                ))}
              </TabsList>


              <div className="px-4" style={{ width: '480px' }}>
                {props.fields.map((field) => (
                  <TabsContent key={field.id} value={field.id} className="mt-0">
                    <div className="px-4 py-2 grid grid-cols-2 text-xs rounded-lg h-full bg-muted">

                      <div className="py-1 col-span-1">
                        <Label htmlFor={field.id} className="font-semibold text-base">
                          {field.label}
                          {field.required && <span className="text-red-500 ml-1">*</span>}
                        </Label>
                      </div>
                      <div className="col-span-1 flex justify-end">
                        {(props.enableEdit|| false)?
                          (<Button 
                            id="toggle-all-descriptions" 
                            variant="outline" 
                            onClick={toggleAllDescriptionsEdit}
                          >
                            {isEditingAllDesc ? "Lock" : "Edit"}
                          </Button>):null}
                      </div>

                      <div className="col-span-2 gap-1.5">
                      {descriptions[field.id]?.map((description, index) => (
                          <div key={index} className="text-xs">
                            {isEditingAllDesc ? (
                              <Input
                                value={description ?? ''}
                                onChange={(e) => handleDescriptionChange(field.id, index, e.target.value)}
                                className="h-7 w-full"
                              />
                            ) : (<p className="block text-muted-foreground w-full p-1">
                                {"• "+description || "No description provided."}</p>)}
                          </div>
                        ))}

                    </div>
                    </div>
                  </TabsContent>
                ))}
              </div>

            <div className="col-span-4 min-w-0 p-6 bg-card flex flex-col justify-between">
              {props.fields.map((field) => (
                  <TabsContent key={field.id} value={field.id} className="mt-0 w-full block">
                    <div className="w-full block">
                        {renderField(field)}
                      </div>
                  </TabsContent>
                ))}  
              </div>
          </Tabs>
        ) : null}
      </CardContent>

      <CardFooter className="col-span-1 flex justify-end gap-2 border-t-2 border-indigo-500 pt-4">
        {/* Global Button to toggle editing across all descriptions */}
        
        <Button id="ticket-reset" variant="ghost" onClick={handleReset}>
          Reset
        </Button>
        <Button id="ticket-cancel" variant="outline" onClick={() => cancelElement?.()}>
          Cancel
        </Button>
        <Button
          id="ticket-submit"
          disabled={!allValid}
          onClick={() => submitProcess()}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  );
}