import glob,pickle, numpy as np

def dump_results(patient="163_A1",scheme="2/1"):
    n_sections_per_slide,n_blocks_per_section=np.array(scheme.split("/")).astype(int)
    images=sorted(glob.glob(f"inputs/{patient}*.npy"))
    image_ids=np.vectorize(lambda x: os.path.basename(x).replace(".npy",""))(images)
    masks=sorted(glob.glob(f"masks/{patient}*macro*.npy"))
    tumor_gnn_results=sorted(glob.glob(f"gnn_results/{patient}*tumor*.pkl"))
    macro_gnn_results=sorted(glob.glob(f"gnn_results/{patient}*macro*.pkl"))
    quality_scores=sorted(glob.glob(f"quality_scores/{patient}*.pkl"))
    ink_results=sorted(glob.glob(f"detected_inks/{patient}*.pkl"))
    nuclei_results=sorted(glob.glob(f"nuclei_results/{patient}*.npy"))

    pickle.dump(dict(n_slides=len(images),
                    image_ids=image_ids,
                    n_sections_per_slide=n_sections_per_slide,
                    n_blocks_per_section=n_blocks_per_section,
                    images=images,
                    masks=masks,
                    tumor_gnn_results=tumor_gnn_results,
                    macro_gnn_results=macro_gnn_results,
                    quality_scores=quality_scores,
                    ink_results=ink_results,
                    nuclei_results=nuclei_results),open(f'results/{patient}.pkl','wb'))
