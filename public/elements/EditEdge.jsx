import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useMemo, useState } from 'react';

export default function EditEdge(props) {
  const [timeLeft, setTimeLeft] = useState(props.timeout || 30);
  const [values, setValues] = useState(() => {
    const init = {};
    (props.fields || []).forEach((f) => {
      init[f.id] = f.value || '';
    });
    return init;
  });

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

  const handleReset = () => {
    const init = {};
    (props.fields || []).forEach((f) => {
      init[f.id] = f.value || '';
    });
    setValues(init);
  };


  return (
    <Card id="jira-ticket" className="mt-4 w-full max-w-3xl grid grid-cols-2 gap-4">
      <CardHeader className="col-span-2 space-y-1">
        <p className="text-sm font-medium text-muted-foreground">
          {"Please Submit new Information!"}
        </p>
      </CardHeader>
      <CardContent className="col-span-2 grid grid-cols-2 gap-4">
        <span>{props.text}</span>
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
          onClick={() => props.onSubmit?.(values)}
        >
          Submit
        </Button>
      </CardFooter>
    </Card>
  );
}