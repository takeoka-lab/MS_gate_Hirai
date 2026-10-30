import fs from 'node:fs/promises';
import path from 'node:path';
import {finalizePresentation} from '/Users/hirainoa/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Users/hirainoa/Desktop/量子/project/Moon_shot__高橋PJ/MS_gate_Hirai';
const build=path.join(root,'output/noise_correlation_20261027/pptx_build');
const skill='/Users/hirainoa/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const finalPath=path.join(root,'output/powerpoint/noise_correlation_atlas_editable_20261027.pptx');
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const chartOwners=Array.from({length:26},(_,i)=>i+3).filter(i=>i<7||i>12);
const tableOwners=[7,8,9,10,11,12,18,19,20,21,22];
const result=await finalizePresentation({
  workspaceDir:root,candidatePath:path.join(build,'candidate.pptx'),finalPath,
  pythonExecutable:'/Users/hirainoa/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  explicitTotalSlideCount:28,requiredNativeChartOwnerSlides:chartOwners,requiredNativeTableOwnerSlides:tableOwners,
  materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:'powerpoint',
  layoutArgs:['--expected-slide-size-emu',`${Math.round(1600*9525)},${Math.round(1131.4*9525)}`,'--validate-heading-fit','--approved-dense-slide','1','--approved-dense-slide','2'],
  fontPolicy:{basis:'design',families:['Arial Unicode MS']},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'final.validation.json'),
});
console.log(JSON.stringify(result));
