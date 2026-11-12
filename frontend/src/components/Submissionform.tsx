"use client";

import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { API_ENDPOINTS } from "@/config/api";

export default function SubmissionForm() {
  const [hackathonId, setHackathonId] = useState("");
  const [projectTitle, setProjectTitle] = useState("");
  const [githubUrl, setGithubUrl] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [pptUrl, setPptUrl] = useState("");
  const [notes, setNotes] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);

    const submissionData = {
      hackathon_id: hackathonId || undefined,
      project_name: projectTitle,
      github_link: githubUrl,
      video_link: videoUrl,
      ppt_link: pptUrl,
      readme_text: notes,
    };

    try {
      const response = await fetch(API_ENDPOINTS.submit, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(submissionData),
      });

      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || "Something went wrong");
      }

      const conf = result.evaluation?.acceptance_probability ?? "N/A";
      alert(`Submitted! Status: ${result.status} (ML Confidence: ${conf}%)`);
    } catch (error) {
      console.error("Submission failed:", error);
      alert(`Submission failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Submit your project</CardTitle>
        <CardDescription>Fill out the details below for ML-assisted evaluation.</CardDescription>
      </CardHeader>
      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="hackathon-id">Session / Hack ID (optional)</Label>
            <Input
              id="hackathon-id"
              placeholder="e.g. 1"
              value={hackathonId}
              onChange={(e) => setHackathonId(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="project-title">Project title</Label>
            <Input
              id="project-title"
              placeholder="e.g. AutoEval"
              value={projectTitle}
              onChange={(e) => setProjectTitle(e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="github-url">GitHub repository URL</Label>
            <Input
              id="github-url"
              placeholder="https://github.com/org/repo"
              value={githubUrl}
              onChange={(e) => setGithubUrl(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="video-url">Video demo URL (optional)</Label>
            <Input
              id="video-url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="ppt-url">Presentation (PPT/Slides) URL (optional)</Label>
            <Input
              id="ppt-url"
              placeholder="https://docs.google.com/presentation/..."
              value={pptUrl}
              onChange={(e) => setPptUrl(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="notes">Project Abstract / README Notes</Label>
            <Textarea
              id="notes"
              placeholder="Describe problem, architecture, ML methodology, and setup..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </div>
        </CardContent>
        <CardFooter>
          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? "Submitting..." : "Submit Project"}
          </Button>
        </CardFooter>
      </form>
    </Card>
  );
}
