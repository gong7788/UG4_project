(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(clear b3)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b5 b6)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(on b3 b4)
		(in-tower b3)
		(green b0)
		(maroon b0)
		(blue b1)
		(green b1)
		(maroon b1)
		(red b2)
		(blue b2)
		(green b2)
		(maroon b2)
		(blue b3)
		(green b3)
		(maroon b3)
		(blue b0)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (not (on b2 b3))))
)