import { useRef, useState } from "react";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Upload, FileUp, LoaderCircle } from "lucide-react";
import { client, queryClient, errorMessage, fileSize } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { ErrorMessage } from "@/components/feedback";

const schema = z.object({
  title: z.string().trim().min(1, "Give the dataset a title.").max(200),
  file: z
    .instanceof(File, { error: "Choose a CSV file." })
    .refine(
      (file) => file.size > 0 && file.size <= 10 * 1024 * 1024,
      "Choose a nonempty file up to 10 MB.",
    )
    .refine(
      (file) => file.name.toLowerCase().endsWith(".csv"),
      "Choose a .csv file.",
    ),
});
type Values = z.infer<typeof schema>;

function UploadForm({ close }: { close: () => void }) {
  const form = useForm<Values>({
    resolver: zodResolver(schema),
    defaultValues: { title: "" },
  });
  const submission = useRef<{ title: string; file: File; id: string } | null>(
    null,
  );
  const navigate = useNavigate();
  const upload = useMutation({
    mutationFn: async ({ title, file }: Values) => {
      if (
        submission.current?.file !== file ||
        submission.current.title !== title
      )
        submission.current = { title, file, id: crypto.randomUUID() };
      const body = {
        title,
        submission_id: submission.current.id,
        file: file.name,
      };
      const result = await client.POST("/api/datasets", {
        body,
        bodySerializer: () => {
          const multipart = new FormData();
          multipart.set("title", title);
          multipart.set("submission_id", body.submission_id);
          multipart.set("file", file);
          return multipart;
        },
      });
      if (result.error) throw new Error(errorMessage(result.error));
      return result.data;
    },
    onSuccess: async (dataset) => {
      await queryClient.invalidateQueries({
        queryKey: ["get", "/api/datasets"],
      });
      close();
      await navigate({
        to: "/datasets/$datasetId",
        params: { datasetId: dataset.id },
        search: { page: 1 },
      });
    },
  });
  return (
    <DialogContent
      showCloseButton={!upload.isPending}
      onEscapeKeyDown={(event) => {
        if (upload.isPending) event.preventDefault();
      }}
      onInteractOutside={(event) => {
        if (upload.isPending) event.preventDefault();
      }}
    >
      <DialogTitle className="text-lg font-normal">Add a dataset</DialogTitle>
      <DialogDescription className="text-xs text-muted-foreground">
        Start with a CSV. Keep the source, inspect the evidence.
      </DialogDescription>
      <form
        onSubmit={form.handleSubmit((data) => upload.mutate(data))}
        noValidate
      >
        <div className="form-field">
          <Label htmlFor="dataset-title">Dataset title</Label>
          <Input
            id="dataset-title"
            placeholder="e.g. Student performance"
            maxLength={200}
            autoComplete="off"
            {...form.register("title")}
            aria-invalid={!!form.formState.errors.title}
            aria-describedby={
              form.formState.errors.title ? "title-error" : undefined
            }
            disabled={upload.isPending}
          />
          {form.formState.errors.title ? (
            <span id="title-error" className="field-error">
              {form.formState.errors.title.message}
            </span>
          ) : null}
        </div>
        <Controller
          name="file"
          control={form.control}
          render={({ field, fieldState }) => (
            <div className="form-field">
              <Label htmlFor="dataset-file">CSV file</Label>
              <div className="upload-zone">
                <FileUp size={27} aria-hidden="true" />
                <span className="text-xs">Choose your source data</span>
                <input
                  id="dataset-file"
                  className="file-input"
                  type="file"
                  accept=".csv,text/csv"
                  onBlur={field.onBlur}
                  ref={field.ref}
                  name={field.name}
                  onChange={(event) => field.onChange(event.target.files?.[0])}
                  disabled={upload.isPending}
                  aria-invalid={!!fieldState.error}
                  aria-describedby="file-help file-error"
                />
                <span
                  id="file-help"
                  className="text-[10px] text-muted-foreground"
                >
                  UTF-8 CSV · up to 10 MB
                </span>
                {field.value ? (
                  <span className="text-[10px]">
                    {fileSize(field.value.size)} selected
                  </span>
                ) : null}
              </div>
              <span id="file-error" className="field-error">
                {fieldState.error?.message}
              </span>
            </div>
          )}
        />
        <p className="subtext text-[11px]">
          The original file is preserved. Saving creates its first version.
        </p>
        {upload.error ? (
          <div className="mt-4">
            <ErrorMessage error={upload.error} />
          </div>
        ) : null}
        <div className="modal-actions">
          <Button
            variant="outline"
            type="button"
            onClick={close}
            disabled={upload.isPending}
          >
            Cancel
          </Button>
          <Button type="submit" disabled={upload.isPending}>
            {upload.isPending ? (
              <LoaderCircle className="animate-spin" aria-hidden="true" />
            ) : (
              <Upload size={14} aria-hidden="true" />
            )}
            {upload.isPending ? "Saving dataset..." : "Save dataset"}
          </Button>
        </div>
      </form>
    </DialogContent>
  );
}

export function UploadDialog({ label = "Upload CSV" }: { label?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Upload size={14} aria-hidden="true" />
          {label}
        </Button>
      </DialogTrigger>
      {open ? <UploadForm close={() => setOpen(false)} /> : null}
    </Dialog>
  );
}
