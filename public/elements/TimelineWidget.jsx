import React, { useRef, useEffect, useState } from "react";

function DownwardArrow(divHeight) {
  return (
    <div className="absolute border-2 w-0.25 bg-gray-800 mx-2 inline-block align-middle" style={{ left: '-27px', top: '18px', height:'60px' }}>
      
    </div>
  );
}

export default function TimelineWidget() {
  const steps = props.steps || [];
  const divRef = useRef(null);
  const [divHeight, setDivHeight] = useState(0);

  useEffect(() => {
    if (divRef.current) {
      // Get the exact height in pixels
      const height = divRef.current.getBoundingClientRect().height;
      setDivHeight(height);
    }
  }, []);

  return (
    <div className="w-full p-4 bg-none rounded-lg shadow-sm">
      {/* Parent container with relative positioning so absolute children align to it */}
      <div className="relative pl-8 flex flex-col" style={{rowGap: '24px'}}>
      

        {steps.map((step, index) => (
          <div key={step.id || index} className="flex flex-row flex-wrap items-start relative">
            
            {/* The circle positioned reliably using inline styles */}
            <div 
              className="absolute h-4 w-4 rounded-full bg-foreground border-white shadow-sm z-10 shrink-0"
              style={{ left: '-25px', top: '4px', backgroundColor: '#434242' }}
            />
            
            {/* Content box */}
            <div ref={divRef} className="w-full flex-col">
              <span className="text-base font-bold text-gray-800">{step.title}</span>
              <p className="text-sm text-gray-600 whitespace-pre-line mt-1">
                {step.subtitle}
              </p>
            </div>
            {index < steps.length - 1 && DownwardArrow(divHeight)}
          </div>
          
          
        ))}
      </div>
    </div>
  );
}