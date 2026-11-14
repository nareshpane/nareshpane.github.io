// The original validated transport calculation, separated from presentation.
export function riverSimulator({location=.4,intensity=.5,velocity=.35,dispersion=.005,nx=61,final=1.2}={}){
 const dx=1/(nx-1),steps=Math.ceil(final*(velocity/dx+2*dispersion/dx**2)/.75),dt=final/steps,A=velocity*dt/dx,B=dispersion*dt/dx**2;
 if(!(A+2*B<=.7500000001)||!Number.isFinite(steps)||steps>10000)throw Error('Invalid or excessive river grid');
 const x=Array.from({length:nx},(_,i)=>i*dx),source=x.map(v=>intensity*Math.exp(-.5*((v-location)/.04)**2)/(.04*Math.sqrt(2*Math.PI)));let c=Array(nx).fill(0),field=[c];
 for(let step=0;step<steps;step++){const next=Array(nx).fill(0);for(let j=1;j<nx-1;j++)next[j]=(1-A-2*B)*c[j]+(A+B)*c[j-1]+B*c[j+1]+dt*source[j];c=next;field.push(c)}
 return{x,t:Array.from({length:steps+1},(_,i)=>i*dt),field,combined:A+2*B};
}
export function sensorAt(sim,time,index){const position=time/sim.t.at(-1)*(sim.t.length-1),i=Math.min(sim.t.length-2,Math.floor(position)),fraction=position-i;return sim.field[i][index]*(1-fraction)+sim.field[i+1][index]*fraction}
export function observationLoss(sim,data){let sum=0,count=0;for(const row of data.train_observation_rows)for(let k=0;k<data.sensors.length;k++){const residual=sensorAt(sim,data.t[data.observation_indices[row]],data.sensors[k])-data.observations[row][k];sum+=residual*residual;count++}return sum/count/(2*data.config.noise_sd**2)}
