"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { format } from "date-fns";
import React, { useEffect, useRef, useState } from "react";
import { toast } from "sonner";


type Campaign={
    name:string;
    uploadDate:string;
    campaignId:string;
    key:string;
}
export default function UploadPage() {
    const [file, setFile] = useState<File | null>(null);
    const [campaigns, setCampaigns] = useState<Campaign[]>([]);
    
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    const fetchCampaigns = async ()=>{
        try {
            const res= await fetch("/api/campaigns");

            if(!res.ok) throw new Error("failed to fetch campaigns!");

            const data= await res.json();
            const campaignArray:Campaign[]=data.campaigns;
            setCampaigns(campaignArray);

            console.log(campaignArray)
            
        } catch (error) {
            toast.error("Error loading campaigns!")
            
        }
    }

    useEffect(()=>{
        fetchCampaigns();
    },[]);

    const handleFileChange = (e:React.ChangeEvent<HTMLInputElement>)=>{
        const file= e.target.files?.[0];
        if(file) setFile(file)
    }



    const handleUpload = async() => {

        if(!file) return;

        const formData= new FormData();
        formData.append("file",file);
        try {
            const res= await fetch("/api/upload",{
                method:"POST",
                body:formData,
            })

            if(!res.ok) throw new Error("Upload failed!");

            const result= await res.json();

            toast.success("Campaign added successfully!", {
                description: `${format(new Date(),"EEEE, MMMM dd, yyyy 'at' h:mm a")}`,
            });

            setFile(null);
            if(fileInputRef.current){
                  fileInputRef.current.value="";
            }
           
            fetchCampaigns();

            
        } catch (error) {
            toast.error("Upload failed!")
            console.error("Upload error:",error)
            
        }
    }

    return (
        <div className="max-w-3xl mx-auto py-10 px-4">
            <h1 className="text-2xl font-bold mb-6">Upload New Campaign</h1>

            <div className="space-y-4 mb-8 ">
                <Input type="file" accept=".csv,.json" onChange={handleFileChange} disabled={!!file} className="block w-full border border-gray-300 p-2 rounded" ref={fileInputRef} />
                {file && (
                    <Card>
                        <CardContent className="p-4">
                            <p>
                                <strong>File: </strong>
                                {file.name}

                            </p>

                            <p>
                                <strong>Size: </strong>
                                {(file.size / 1024).toFixed(2)} KB
                            </p>
                            <Button className="mt-4" onClick={handleUpload}>Upload</Button>
                        </CardContent>

                    </Card>
                )}
            </div>

            <h2 className="text-xl font-semibold mb-4">Recent Uploads</h2>
            <div className="space-y-4">
                {
                    campaigns.map((campaign,index)=>(
                        <Card key={index}>
                            <CardContent className="p-4">
                                <p><strong>{campaign.name}</strong></p>
                                <p>Uploaded At: {format(new Date(campaign.uploadDate),'EEEE,MMMM dd,yyyy \'at\' h:mm a')}</p>
                            </CardContent>
                        </Card>
                    ))
                }

                {
                    campaigns.length===0 && (
                        <p className="text-sm text-gray-500">No Uploads Yet</p>
                    )
                }
              
              
            </div>

        </div>

    )


}